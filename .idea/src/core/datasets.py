# 双模态生物特征数据集定义
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import random


class FaceDataset(Dataset):
    """人脸识别数据集类"""

    def __init__(self, data_dir, mode='train', transform=None, image_size=224):
        """
        Args:
            data_dir: 数据根目录路径
            mode: 数据集模式 ('train', 'val', 'test')
            transform: 数据变换
            image_size: 图像大小
        """
        self.data_dir = data_dir
        self.mode = mode
        self.image_size = image_size

        # 如果没有提供transform，根据模式创建默认的
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform

        # 收集所有图像路径和标签
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        self._load_data()

    def _load_data(self):
        """加载数据文件路径和标签"""
        if not os.path.exists(self.data_dir):
            raise ValueError(f"数据目录不存在: {self.data_dir}")

        # 假设数据按类别文件夹组织
        class_names = sorted(os.listdir(self.data_dir))
        class_names = [name for name in class_names if os.path.isdir(os.path.join(self.data_dir, name))]

        for idx, class_name in enumerate(class_names):
            self.class_to_idx[class_name] = idx
            class_dir = os.path.join(self.data_dir, class_name)

            # 获取该类别下的所有图像文件
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(idx)

        print(f"加载了 {len(self.image_paths)} 张图像，{len(self.class_to_idx)} 个类别")
        print(f"数据目录: {self.data_dir}")
        print(f"类别列表: {list(self.class_to_idx.keys())[:5]}...")
        if self.image_paths:
            print(f"示例图像路径: {self.image_paths[0]}")

    def _get_default_transform(self):
        """获取默认的数据变换"""
        if self.mode == 'train':
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:  # val/test
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # 加载图像
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"加载图像失败 {img_path}: {e}")
            # 返回一个空白图像作为替代
            image = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'path': img_path
        }


class BiometricDataset(Dataset):
    """双模态生物特征数据集类（预留接口）"""

    def __init__(self, face_paths, finger_paths, labels, transform=None):
        """
        Args:
            face_paths: 人脸图像路径列表
            finger_paths: 指纹图像路径列表
            labels: 对应的身份标签
            transform: 数据增强变换
        """
        self.face_paths = face_paths
        self.finger_paths = finger_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # TODO: 实现双模态数据加载
        pass