# 模型模块初始化文件
from .face_net import FaceNet, FaceNetWithArcFace, create_face_model
from .fingerprint_net import FingerprintNet, create_fingerprint_model
from .fusion_model import FusionModel, create_fusion_model


def create_model(modality: str = "face", **kwargs):
    """Simple factory that dispatches to modality implementations."""
    if modality.lower() == "face":
        return create_face_model(**kwargs)
    elif modality.lower() == "fingerprint":
        return create_fingerprint_model(**kwargs)
    elif modality.lower() == "fusion":
        return create_fusion_model(**kwargs)
    else:
        raise ValueError(f"Unknown modality: {modality}")


__all__ = [
    "FaceNet",
    "FaceNetWithArcFace",
    "create_face_model",
    "FingerprintNet",
    "create_fingerprint_model",
    "FusionModel",
    "create_fusion_model",
    "create_model",
]