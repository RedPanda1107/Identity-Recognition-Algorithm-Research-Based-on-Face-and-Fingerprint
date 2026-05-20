"""
指纹推理模块。

提供 FingerprintInferencer 类用于从指纹图像提取特征向量。
包含 CLAHE 预处理，确保与训练一致。
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger("FingerprintInferencer")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FingerprintPreprocessor:
    """指纹图像预处理器（推理用）

    流程：灰度转换 → CLAHE 增强 → RGB 合并 → ImageNet 归一化。
    与训练时的 FusionDataset 和 FingerprintDataset 完全一致。
    """

    def __init__(self, image_size: int = 224, use_clahe: bool = True,
                 clahe_clip_limit: float = 2.0, clahe_tile_size=(8, 8)):
        self.image_size = image_size
        self.use_clahe = use_clahe
        if use_clahe:
            self._clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_size)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _apply_clahe(self, img: Image.Image) -> Image.Image:
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        enhanced = self._clahe.apply(gray)
        return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if self.use_clahe:
            image = self._apply_clahe(image)
        return self.transform(image)

    def preprocess_batch(self, images: list[Image.Image]) -> torch.Tensor:
        tensors = [self.preprocess(img) for img in images]
        return torch.stack(tensors, dim=0)


class FingerprintInferencer:
    """指纹特征提取推理器"""

    def __init__(self, model_loader):
        self.loader = model_loader
        self._model = None
        self._preprocessor = None

    @property
    def model(self):
        if self._model is None:
            self._model = self.loader.load_fingerprint_model()
        return self._model

    @property
    def preprocessor(self):
        if self._preprocessor is None:
            self._preprocessor = FingerprintPreprocessor(image_size=224, use_clahe=True)
        return self._preprocessor

    def preprocess_image(self, image) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image = Image.open(str(image)).convert("RGB")
        elif isinstance(image, torch.Tensor):
            return image.unsqueeze(0) if image.dim() == 3 else image
        tensor = self.preprocessor.preprocess(image)
        return tensor.unsqueeze(0)

    def extract(self, image) -> torch.Tensor:
        tensor = self.preprocess_image(image)
        with torch.no_grad():
            emb = self.model.extract_features(tensor)
            emb = F.normalize(emb, p=2, dim=1)
        return emb.squeeze().cpu().numpy()
