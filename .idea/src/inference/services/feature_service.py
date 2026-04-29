"""
特征提取服务 - 统一封装人脸/指纹/融合三种特征提取接口
"""

import logging
from typing import Literal, Optional

import numpy as np

project_root = __file__
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ..model_loader import ModelLoader
from ..face_inference import FaceInferencer
from ..fingerprint_inference import FingerprintInferencer
from ..fusion_inference import FusionInferencer

logger = logging.getLogger("FeatureService")


class FeatureService:
    """统一特征提取服务

    提供三种模态的特征提取：
    - face:       人脸特征提取
    - fingerprint: 指纹特征提取
    - fusion:     人脸+指纹融合特征提取
    """

    def __init__(
        self,
        device: str = "auto",
        checkpoint_dir: Optional[str] = None,
        num_classes: int = 500,
    ):
        self.model_loader = ModelLoader(
            device=device,
            checkpoint_dir=checkpoint_dir,
            num_classes=num_classes,
        )
        self._face: Optional[FaceInferencer] = None
        self._fingerprint: Optional[FingerprintInferencer] = None
        self._fusion: Optional[FusionInferencer] = None

    @property
    def face(self) -> FaceInferencer:
        if self._face is None:
            self._face = FaceInferencer(self.model_loader)
        return self._face

    @property
    def fingerprint(self) -> FingerprintInferencer:
        if self._fingerprint is None:
            self._fingerprint = FingerprintInferencer(self.model_loader)
        return self._fingerprint

    @property
    def fusion(self) -> FusionInferencer:
        if self._fusion is None:
            self._fusion = FusionInferencer(self.model_loader)
        return self._fusion

    def extract(
        self,
        image,
        modality: Literal["face", "fingerprint"] = "face",
        face_image=None,
        fp_image=None,
        fusion_method: Literal["simple", "adaptive", "gated", "hierarchical"] = "simple",
    ) -> np.ndarray:
        """统一特征提取接口

        单模态用法:
            extract(image, modality="face")
            extract(image, modality="fingerprint")

        融合用法:
            extract(None, face_image=img1, fp_image=img2, fusion_method="simple")
        """
        if modality == "face" and face_image is None:
            return self.face.extract(image)

        if modality == "fingerprint" and fp_image is None:
            return self.fingerprint.extract(image)

        if face_image is not None and fp_image is not None:
            result = self.fusion.extract_all(face_image, fp_image, method=fusion_method)
            return result["fused_embedding"]

        raise ValueError("Invalid arguments: must specify either modality or both face_image and fp_image")

    def extract_face(self, image) -> np.ndarray:
        """提取人脸特征"""
        return self.face.extract(image)

    def extract_fingerprint(self, image) -> np.ndarray:
        """提取指纹特征"""
        return self.fingerprint.extract(image)

    def extract_fusion(
        self,
        face_image,
        fp_image,
        method: Literal["simple", "adaptive", "gated", "hierarchical"] = "simple",
    ) -> np.ndarray:
        """提取融合特征"""
        result = self.fusion.extract_all(face_image, fp_image, method=method)
        return result["fused_embedding"]

    def extract_all_modalities(
        self,
        face_image,
        fp_image,
        method: Literal["simple", "adaptive", "gated", "hierarchical"] = "simple",
    ) -> dict:
        """同时提取人脸、指纹和融合特征"""
        return self.fusion.extract_all(face_image, fp_image, method=method)

    def unload_models(self):
        """卸载所有模型，释放显存"""
        self.model_loader.unload_all()
