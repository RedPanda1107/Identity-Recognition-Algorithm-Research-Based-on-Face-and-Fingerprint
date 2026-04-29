"""
人脸推理模块 - 图像预处理 + 特征提取
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

logger = logging.getLogger("FaceInferencer")


class FaceInferencer:
    """人脸特征提取推理器"""

    # 标准人脸图像尺寸（ImageNet / ArcFace 常用）
    IMAGE_SIZE = 224

    # ImageNet 标准化参数
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, model_loader: ModelLoader):
        self.loader = model_loader
        self._model: torch.nn.Module = None

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            self._model = self.loader.load_face_model()
        return self._model

    def preprocess_image(self, image: Union[np.ndarray, Image.Image, str]) -> torch.Tensor:
        """将输入图像预处理为模型输入张量

        Args:
            image: PIL Image / numpy array (BGR or RGB) / 文件路径

        Returns:
            Tensor [1, 3, H, W]，已归一化
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            if image.shape[-1] == 3 and image.dtype == np.uint8:
                if image.max() > 1:
                    pass
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                if len(gray.shape) == 2:
                    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        image = cv2.resize(image, (self.IMAGE_SIZE, self.IMAGE_SIZE))
        image = image.astype(np.float32) / 255.0

        for i in range(3):
            image[:, :, i] = (image[:, :, i] - self.MEAN[i]) / self.STD[i]

        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor

    @torch.no_grad()
    def extract(self, image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """提取单张人脸图像的 512 维特征向量

        Args:
            image: 输入图像

        Returns:
            归一化后的 512 维特征向量（numpy array）
        """
        tensor = self.preprocess_image(image).to(self.loader.device)
        features = self.model.extract_features(tensor)
        embedding = F.normalize(features, p=2, dim=1).squeeze().cpu().numpy()
        return embedding

    @torch.no_grad()
    def extract_batch(self, images: List[Union[np.ndarray, Image.Image, str]]) -> np.ndarray:
        """批量提取多张人脸图像的特征向量

        Args:
            images: 图像列表

        Returns:
            [N, 512] 归一化特征矩阵
        """
        tensors = [self.preprocess_image(img) for img in images]
        batch = torch.cat(tensors, dim=0).to(self.loader.device)
        features = self.model.extract_features(batch)
        embeddings = F.normalize(features, p=2, dim=1).cpu().numpy()
        return embeddings

    def extract_from_base64(self, base64_str: str) -> np.ndarray:
        """从 Base64 编码的字符串提取人脸特征"""
        import base64
        img_bytes = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return self.extract(img)

    def get_embedding_dim(self) -> int:
        return 512
