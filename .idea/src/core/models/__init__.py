# 模型模块初始化文件
from .face_net import FaceNet, create_face_model
from .fingerprint_net import FingerprintNet, create_fingerprint_model
from .fusion_model import (
    SimpleFusionModel,
    AdaptiveFusionModel,
    GatedFusionModel,
    HierarchicalFusionModel,
    create_fusion_model,
)


def create_model(modality: str = "face", **kwargs):
    """工厂函数：创建模型

    Args:
        modality: 'face', 'fingerprint', 'fusion'
        **kwargs: 传递给模型的参数

    Returns:
        模型实例
    """
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
    "create_face_model",
    "FingerprintNet",
    "create_fingerprint_model",
    "SimpleFusionModel",
    "AdaptiveFusionModel",
    "GatedFusionModel",
    "HierarchicalFusionModel",
    "create_fusion_model",
    "create_model",
]
