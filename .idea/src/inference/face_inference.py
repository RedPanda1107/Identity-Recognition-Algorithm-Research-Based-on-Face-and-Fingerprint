"""
人脸推理模块。

提供 FaceInferencer 类用于从人脸图像提取特征向量。
依赖 ModelLoader 加载预训练模型。
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger("FaceInferencer")


class FaceInferencer:
    """人脸特征提取推理器"""

    def __init__(self, model_loader):
        self.loader = model_loader
        self._model = None
        self._preprocessor = None

    @property
    def model(self):
        if self._model is None:
            self._model = self.loader.load_face_model()
        return self._model

    @property
    def preprocessor(self):
        if self._preprocessor is None:
            from .face_preprocessor import FacePreprocessor
            self._preprocessor = FacePreprocessor(image_size=224)
        return self._preprocessor

    def preprocess_image(self, image) -> torch.Tensor:
        """预处理图像，返回归一化 tensor [1, 3, 224, 224]"""
        if isinstance(image, (str, Path)):
            image = Image.open(str(image)).convert("RGB")
        elif isinstance(image, torch.Tensor):
            t = image.unsqueeze(0) if image.dim() == 3 else image
            return t.to(self.loader.device)
        tensor = self.preprocessor.preprocess(image)
        return tensor.unsqueeze(0).to(self.loader.device)

    def extract(self, image) -> torch.Tensor:
        """从图像提取归一化 512-d 特征向量"""
        tensor = self.preprocess_image(image)
        with torch.no_grad():
            emb = self.model.extract_features(tensor)
            emb = F.normalize(emb, p=2, dim=1)
        return emb.squeeze().cpu().numpy()
