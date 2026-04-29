"""
FastAPI 依赖注入
"""

import logging
from functools import lru_cache
from typing import Optional

from .config import (
    DEVICE, MODEL_CHECKPOINT_DIR, MODEL_NUM_CLASSES,
    MODEL_EMBEDDING_DIM, MODEL_FUSION_DIM, GALLERY_DIR,
)

logger = logging.getLogger("Dependencies")


@lru_cache()
def get_feature_service():
    """获取特征提取服务实例（单例，缓存）"""
    from ..services.feature_service import FeatureService
    service = FeatureService(
        device=DEVICE,
        checkpoint_dir=MODEL_CHECKPOINT_DIR,
        num_classes=MODEL_NUM_CLASSES,
    )
    logger.info("[Dependency] FeatureService created")
    return service


@lru_cache()
def get_matching_service():
    """获取匹配服务实例（单例）"""
    from ..services.matching_service import MatchingService
    service = MatchingService()
    logger.info("[Dependency] MatchingService created")
    return service


@lru_cache()
def get_gallery_manager():
    """获取 Gallery 管理器实例（单例）"""
    from ..gallery_manager import GalleryManager
    manager = GalleryManager(gallery_dir=GALLERY_DIR, auto_save=True)
    logger.info(f"[Dependency] GalleryManager created, users={manager.count_users()}")
    return manager
