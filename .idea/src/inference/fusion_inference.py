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

    支持三种融合策略：
    - simple:    加权求和（可学习权重）
    - adaptive:  注意力自适应融合
    - gated:    门控融合（论文常用方案）
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
        method: Literal["simple", "adaptive", "gated", "hierarchical"]
    ) -> torch.nn.Module:
        if method not in self._fusion_models:
            self._fusion_models[method] = self.loader.load_fusion_model(method=method)
        return self._fusion_models[method]

    def extract_separate(
        self,
        face_image,
        fp_image,
        method: Literal["simple", "adaptive", "gated"] = "simple",
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
    def extract_fused(
        self,
        face_image,
        fp_image,
        method: Literal["simple", "adaptive", "gated", "hierarchical"] = "simple",
    ) -> np.ndarray:
        """提取融合特征向量（512 维）

        Args:
            face_image: 人脸图像
            fp_image: 指纹图像
            method: 融合策略

        Returns:
            归一化融合特征向量 [256] 或 [512]（取决于 fusion_dim）
        """
        face_tensor = self.face_inferencer.preprocess_image(face_image).to(self.loader.device)
        fp_tensor = self.fp_inferencer.preprocess_image(fp_image).to(self.loader.device)

        face_emb = self.face_inferencer.model.extract_features(face_tensor)
        fp_emb = self.fp_inferencer.model.extract_features(fp_tensor)

        fusion_model = self._get_fusion_model(method)
        fused = fusion_model.extract_fused_features(face_emb, fp_emb)
        fused = F.normalize(fused, p=2, dim=1).squeeze().cpu().numpy()
        return fused

    def extract_all(
        self,
        face_image,
        fp_image,
        method: Literal["simple", "adaptive", "gated", "hierarchical"] = "simple",
    ) -> dict:
        """同时提取人脸特征、指纹特征和融合特征

        Returns:
            {
                "face_embedding":  [512] numpy,
                "fp_embedding":    [512] numpy,
                "fused_embedding": [256/512] numpy,
                "face_tensor":      torch.Tensor,
                "fp_tensor":       torch.Tensor,
            }
        """
        face_tensor = self.face_inferencer.preprocess_image(face_image).to(self.loader.device)
        fp_tensor = self.fp_inferencer.preprocess_image(fp_image).to(self.loader.device)

        face_emb = self.face_inferencer.model.extract_features(face_tensor)
        fp_emb = self.fp_inferencer.model.extract_features(fp_tensor)

        fusion_model = self._get_fusion_model(method)
        fused = fusion_model.extract_fused_features(face_emb, fp_emb)
        fused = F.normalize(fused, p=2, dim=1)

        return {
            "face_embedding": F.normalize(face_emb, p=2, dim=1).squeeze().cpu().numpy(),
            "fp_embedding": F.normalize(fp_emb, p=2, dim=1).squeeze().cpu().numpy(),
            "fused_embedding": fused.squeeze().cpu().numpy(),
        }
