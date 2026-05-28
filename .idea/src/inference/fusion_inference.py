"""
融合推理模块 - 人脸 + 指纹联合特征提取与匹配
"""

import logging
import sys
from pathlib import Path
from typing import Union, List, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from .model_loader import ModelLoader
from .face_inference import FaceInferencer
from .fingerprint_inference import FingerprintInferencer

logger = logging.getLogger("FusionInferencer")


class FusionInferencer:
    """融合特征提取推理器（人脸 + 指纹）

    支持两种融合策略：
    - simple:    加权求和（可学习权重）
    - adaptive:  注意力自适应融合
    """

    def __init__(self, model_loader: ModelLoader):
        self.loader = model_loader
        self._face = None
        self._fingerprint = None
        self._fusion_models: dict = {}

    @property
    def face_inferencer(self) -> FaceInferencer:
        if self._face is None:
            self._face = FaceInferencer(self.loader)
        return self._face

    @property
    def fp_inferencer(self) -> FingerprintInferencer:
        if self._fingerprint is None:
            self._fingerprint = FingerprintInferencer(self.loader)
        return self._fingerprint

    def _get_fusion_model(
        self,
        method: Literal["simple", "adaptive"]
    ) -> torch.nn.Module:
        if method not in self._fusion_models:
            self._fusion_models[method] = self.loader.load_fusion_model(method=method)
        return self._fusion_models[method]

    def extract_separate(
        self,
        face_image,
        fp_image,
        method: Literal["simple", "adaptive"] = "simple",
    ) -> dict:
        """分别提取人脸和指纹特征（不进行融合，用于融合识别前的独立置信度计算）

        Returns:
            {
                "face_embedding": [512], 归一化人脸特征
                "fp_embedding":   [512], 归一化指纹特征
            }
        """
        face_emb = self.face_inferencer.extract(face_image)
        fp_emb = self.fp_inferencer.extract(fp_image)
        return {
            "face_embedding": face_emb,
            "fp_embedding": fp_emb,
        }

    @torch.no_grad()
    def project_to_fusion_space(
        self,
        face_feature: np.ndarray,
        fp_feature: np.ndarray,
        method: Literal["simple", "adaptive"] = "simple",
    ) -> np.ndarray:
        """将单模态或双模态特征投影到融合空间（256-d）

        当缺失某个模态时，传入对应零向量即可。
        融合模型会学习降低缺失模态的权重。
        """
        embedding_dim = self.loader.embedding_dim

        face_t = (
            torch.from_numpy(face_feature).float().to(self.loader.device)
            if face_feature is not None and np.any(face_feature != 0)
            else torch.zeros(1, embedding_dim, device=self.loader.device)
        )
        fp_t = (
            torch.from_numpy(fp_feature).float().to(self.loader.device)
            if fp_feature is not None and np.any(fp_feature != 0)
            else torch.zeros(1, embedding_dim, device=self.loader.device)
        )

        if face_t.dim() == 1:
            face_t = face_t.unsqueeze(0)
        if fp_t.dim() == 1:
            fp_t = fp_t.unsqueeze(0)

        fusion_model = self._get_fusion_model(method)
        fused = fusion_model.extract_fused_features(face_t, fp_t)
        fused = F.normalize(fused, p=2, dim=1).squeeze().cpu().numpy()
        return fused

    @torch.no_grad()
    def project_gallery_to_fusion_space(
        self,
        gallery_features: list[np.ndarray],
        modality: Literal["face", "fingerprint"],
        method: Literal["simple", "adaptive"] = "simple",
    ) -> np.ndarray:
        """批量将 Gallery 特征投影到融合空间

        缺失的模态以零向量替代。
        """
        if not gallery_features:
            return np.array([])

        embedding_dim = self.loader.embedding_dim
        fusion_model = self._get_fusion_model(method)

        feats = np.array([np.asarray(f, dtype=np.float32).flatten() for f in gallery_features])
        feats_t = torch.from_numpy(feats).float().to(self.loader.device)

        if modality == "face":
            zeros_t = torch.zeros(feats_t.shape[0], embedding_dim, device=self.loader.device)
            fused = fusion_model.extract_fused_features(feats_t, zeros_t)
        else:
            zeros_t = torch.zeros(feats_t.shape[0], embedding_dim, device=self.loader.device)
            fused = fusion_model.extract_fused_features(zeros_t, feats_t)

        fused = F.normalize(fused, p=2, dim=1).cpu().numpy()
        return fused

    @torch.no_grad()
    def extract_all(
        self,
        face_image,
        fp_image,
        method: Literal["simple", "adaptive"] = "simple",
    ) -> dict:
        """同时提取人脸特征、指纹特征和融合特征

        支持单模态输入：
            - 仅传人脸 → 指纹以零向量替代
            - 仅传指纹 → 人脸以零向量替代
            - 双模态 → 完整融合

        Returns:
            {
                "face_embedding":  [512] numpy,
                "fp_embedding":    [512] numpy,
                "fused_embedding": [256/512] numpy,
                "modality": "fusion" | "face_only" | "fingerprint_only",
            }
        """
        embedding_dim = self.loader.embedding_dim

        # 人脸特征提取（None 时用零向量）
        if face_image is not None:
            face_tensor = self.face_inferencer.preprocess_image(face_image).to(self.loader.device)
            face_emb_raw = self.face_inferencer.model.extract_features(face_tensor)
        else:
            face_emb_raw = torch.zeros(1, embedding_dim, device=self.loader.device)

        # 指纹特征提取（None 时用零向量）
        if fp_image is not None:
            fp_tensor = self.fp_inferencer.preprocess_image(fp_image).to(self.loader.device)
            fp_emb_raw = self.fp_inferencer.model.extract_features(fp_tensor)
        else:
            fp_emb_raw = torch.zeros(1, embedding_dim, device=self.loader.device)

        # 统一走融合模型
        fusion_model = self._get_fusion_model(method)
        fused = fusion_model.extract_fused_features(face_emb_raw, fp_emb_raw)
        fused = F.normalize(fused, p=2, dim=1)

        # 模态标记
        if face_image is not None and fp_image is not None:
            modality = "fusion"
        elif face_image is not None:
            modality = "face_only"
        else:
            modality = "fingerprint_only"

        return {
            "face_embedding": F.normalize(face_emb_raw, p=2, dim=1).squeeze().cpu().numpy(),
            "fp_embedding": F.normalize(fp_emb_raw, p=2, dim=1).squeeze().cpu().numpy(),
            "fused_embedding": fused.squeeze().cpu().numpy(),
            "modality": modality,
        }
