# 模型模块初始化文件
from .face_net import FaceNet, create_face_model
from .fingerprint_net import FingerprintNet, create_fingerprint_model
from .fusion_model import (
    FusionModel,
    ModalityProjection,
)
from .fusion_strategy import (
    FusionStrategy,
    WeightedSumStrategy,
    AttentionStrategy,
    AblationStrategy,
    create_fusion_strategy,
    STRATEGY_REGISTRY,
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


def create_fusion_model(fusion_strategy='simple', **kwargs):
    """工厂函数：创建融合模型

    Args:
        fusion_strategy: 融合策略类型，'simple' | 'adaptive'
        **kwargs: 传递给 FusionModel 的额外参数

    Returns:
        FusionModel 实例
    """
    return FusionModel(fusion_strategy=fusion_strategy, **kwargs)


# 保留旧类名的别名，确保向后兼容（现有代码中的 create_fusion_model 可能直接引用旧类）
SimpleFusionModel = FusionModel
AdaptiveFusionModel = FusionModel


__all__ = [
    "FaceNet", "create_face_model",
    "FingerprintNet", "create_fingerprint_model",
    "FusionModel", "ModalityProjection",
    "FusionStrategy", "WeightedSumStrategy", "AttentionStrategy",
    "AblationStrategy", "create_fusion_strategy", "STRATEGY_REGISTRY",
    "create_model", "create_fusion_model",
    "SimpleFusionModel", "AdaptiveFusionModel",
]
