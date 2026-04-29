"""
指纹推理模块 - 图像预处理 + 特征提取
"""

import io
import logging
import sys
from pathlib import Path
from typing import Union, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from .model_loader import ModelLoader

logger = logging.getLogger("FingerprintInferencer")


class FingerprintInferencer:
    """指纹特征提取推理器"""

    IMAGE_SIZE = 224

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, model_loader: ModelLoader):
        self.loader = model_loader
        self._model: torch.nn.Module = None

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            self._model = self.loader.load_fingerprint_model()
        return self._model

    def _preprocess_fp(self, image: np.ndarray) -> np.ndarray:
        """指纹图像增强预处理

        步骤：
        1. 灰度化
        2. 直方图均衡化（增强纹路对比度）
        3. 中值滤波（去噪）
        4. 调整亮度和对比度
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        gray = cv2.equalizeHist(gray)

        k = 3
        gray = cv2.medianBlur(gray, k)

        alpha = 1.2
        beta = 10
        gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

        return gray

    def preprocess_image(self, image: Union[np.ndarray, Image.Image, str]) -> torch.Tensor:
        """将输入图像预处理为模型输入张量"""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            if image.shape[-1] == 3:
                gray_check = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                if np.std(gray_check) < 30:
                    image = cv2.cvtColor(gray_check, cv2.COLOR_GRAY2RGB)

        gray = self._preprocess_fp(image)
        gray = cv2.resize(gray, (self.IMAGE_SIZE, self.IMAGE_SIZE))

        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        rgb = rgb.astype(np.float32) / 255.0

        for i in range(3):
            rgb[:, :, i] = (rgb[:, :, i] - self.MEAN[i]) / self.STD[i]

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor

    @torch.no_grad()
    def extract(self, image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """提取单张指纹图像的 512 维特征向量"""
        tensor = self.preprocess_image(image).to(self.loader.device)
        features = self.model.extract_features(tensor)
        embedding = F.normalize(features, p=2, dim=1).squeeze().cpu().numpy()
        return embedding

    @torch.no_grad()
    def extract_batch(self, images: List[Union[np.ndarray, Image.Image, str]]) -> np.ndarray:
        """批量提取指纹特征向量"""
        tensors = [self.preprocess_image(img) for img in images]
        batch = torch.cat(tensors, dim=0).to(self.loader.device)
        features = self.model.extract_features(batch)
        embeddings = F.normalize(features, p=2, dim=1).cpu().numpy()
        return embeddings

    def extract_from_base64(self, base64_str: str) -> np.ndarray:
        """从 Base64 编码的字符串提取指纹特征"""
        import base64
        img_bytes = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return self.extract(img)

    def get_embedding_dim(self) -> int:
        return 512
