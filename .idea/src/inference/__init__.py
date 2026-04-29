"""
Inference 模块 - 门禁系统推理服务
提供人脸、指纹、融合模型的推理能力
"""

from .model_loader import ModelLoader
from .face_inference import FaceInferencer
from .fingerprint_inference import FingerprintInferencer
from .fusion_inference import FusionInferencer
from .gallery_manager import GalleryManager

__all__ = [
    "ModelLoader",
    "FaceInferencer",
    "FingerprintInferencer",
    "FusionInferencer",
    "GalleryManager",
]
