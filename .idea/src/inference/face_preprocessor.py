"""
人脸图像推理预处理模块。

提供独立的图像预处理流程，确保推理时使用与训练完全一致的归一化参数。
"""

import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FacePreprocessor:
    """人脸图像预处理器（推理用）

    使用与训练完全一致的 ImageNet 归一化参数。
    """

    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """预处理 PIL Image，返回归一化 tensor [3, H, W]"""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return self.transform(image)

    def preprocess_batch(self, images: list[Image.Image]) -> torch.Tensor:
        """预处理一批 PIL Image，返回归一化 tensor [B, 3, H, W]"""
        tensors = [self.preprocess(img) for img in images]
        return torch.stack(tensors, dim=0)
