"""
模型加载器 - 统一管理人脸/指纹/融合模型的加载与缓存
"""

import os
import sys
import glob
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Literal

import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from core.models import (
    FaceNet, create_face_model,
    FingerprintNet, create_fingerprint_model,
    SimpleFusionModel, AdaptiveFusionModel, GatedFusionModel, HierarchicalFusionModel,
    create_fusion_model,
)

logger = logging.getLogger("ModelLoader")


class ModelLoader:
    """统一模型加载器，支持加载人脸/指纹/融合预训练模型并缓存实例"""

    FUSION_METHODS = {
        "simple": SimpleFusionModel,
        "adaptive": AdaptiveFusionModel,
        "gated": GatedFusionModel,
        "hierarchical": HierarchicalFusionModel,
    }

    def __init__(self, device: str = "auto", checkpoint_dir: Optional[str] = None,
                 num_classes: int = 500, embedding_dim: int = 512, fusion_dim: int = 256):
        """
        Args:
            device: 计算设备，"auto" / "cuda" / "cpu"
            checkpoint_dir: 检查点根目录，默认 checkpoints/
            num_classes: 分类类别数（需与训练时一致）
            embedding_dim: 特征维度
            fusion_dim: 融合特征维度
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else project_root / "checkpoints"
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.fusion_dim = fusion_dim

        self._cache: Dict[str, torch.nn.Module] = {}
        self._loaded_paths: Dict[str, str] = {}

        logger.info(f"[ModelLoader] device={self.device}, checkpoint_dir={self.checkpoint_dir}")

    def _find_checkpoint(self, pattern: str) -> Optional[str]:
        """在 checkpoint 目录中查找匹配的模型文件"""
        matches = list(self.checkpoint_dir.rglob(pattern))
        if not matches:
            matches = list((project_root / "scripts" / "checkpoints").rglob(pattern))
        return str(matches[0]) if matches else None

    def _load_state_dict(self, path: str, model: torch.nn.Module) -> bool:
        """安全加载 state_dict"""
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
            model.load_state_dict(state, strict=False)
            return True
        except Exception as e:
            logger.warning(f"[ModelLoader] Failed to load {path}: {e}")
            return False

    def load_face_model(self, checkpoint_path: Optional[str] = None) -> FaceNet:
        """加载人脸模型"""
        cache_key = "face"
        if cache_key in self._cache:
            logger.info("[ModelLoader] Face model loaded from cache")
            return self._cache[cache_key]

        model = create_face_model(
            model_type="facenet",
            num_classes=self.num_classes,
            embedding_dim=self.embedding_dim,
            pretrained=False,
        ).to(self.device)
        model.eval()

        if checkpoint_path is None:
            checkpoint_path = self._find_checkpoint("best_face*.pth")
        if checkpoint_path and os.path.exists(checkpoint_path):
            if self._load_state_dict(checkpoint_path, model):
                logger.info(f"[ModelLoader] Face model loaded: {checkpoint_path}")
            else:
                logger.warning("[ModelLoader] Face model loaded without pretrained weights")
        else:
            logger.warning("[ModelLoader] No face checkpoint found, using random weights")

        self._cache[cache_key] = model
        self._loaded_paths[cache_key] = checkpoint_path or "random"
        return model

    def load_fingerprint_model(self, checkpoint_path: Optional[str] = None) -> FingerprintNet:
        """加载指纹模型"""
        cache_key = "fingerprint"
        if cache_key in self._cache:
            logger.info("[ModelLoader] Fingerprint model loaded from cache")
            return self._cache[cache_key]

        model = create_fingerprint_model(
            model_type="fingerprint_net",
            num_classes=self.num_classes,
            embedding_dim=self.embedding_dim,
            pretrained=False,
        ).to(self.device)
        model.eval()

        if checkpoint_path is None:
            checkpoint_path = self._find_checkpoint("best_fingerprint*.pth")
        if checkpoint_path and os.path.exists(checkpoint_path):
            if self._load_state_dict(checkpoint_path, model):
                logger.info(f"[ModelLoader] Fingerprint model loaded: {checkpoint_path}")
            else:
                logger.warning("[ModelLoader] FP model loaded without pretrained weights")
        else:
            logger.warning("[ModelLoader] No fingerprint checkpoint found, using random weights")

        self._cache[cache_key] = model
        self._loaded_paths[cache_key] = checkpoint_path or "random"
        return model

    def load_fusion_model(
        self,
        method: Literal["simple", "adaptive", "gated", "hierarchical"] = "simple",
        checkpoint_path: Optional[str] = None,
    ) -> torch.nn.Module:
        """加载融合模型"""
        cache_key = f"fusion_{method}"
        if cache_key in self._cache:
            logger.info(f"[ModelLoader] Fusion model ({method}) loaded from cache")
            return self._cache[cache_key]

        model = create_fusion_model(
            fusion_method=method,
            face_embedding_dim=self.embedding_dim,
            fingerprint_embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
            fusion_dim=self.fusion_dim,
            use_arcface=False,
        ).to(self.device)
        model.eval()

        if checkpoint_path is None:
            checkpoint_path = self._find_checkpoint(f"best_{method}.pth")
        if checkpoint_path and os.path.exists(checkpoint_path):
            if self._load_state_dict(checkpoint_path, model):
                logger.info(f"[ModelLoader] Fusion model ({method}) loaded: {checkpoint_path}")
            else:
                logger.warning(f"[ModelLoader] Fusion model ({method}) loaded without weights")
        else:
            logger.warning(f"[ModelLoader] No fusion checkpoint found for {method}")

        self._cache[cache_key] = model
        self._loaded_paths[cache_key] = checkpoint_path or "random"
        return model

    def get_loaded_models(self) -> Dict[str, str]:
        """返回已加载模型的路径信息"""
        return dict(self._loaded_paths)

    def unload_all(self):
        """卸载所有缓存的模型，释放显存"""
        self._cache.clear()
        self._loaded_paths.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("[ModelLoader] All models unloaded")
