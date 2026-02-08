import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random
import numpy as np


class FaceDataset(Dataset):
    """Face dataset implementation."""

    def __init__(self, data_dir, mode='train', image_size=224, augment=True):
        """
        Args:
            data_dir: Path to face data directory
            mode: 'train' or 'val'
            image_size: Image size for resizing
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.mode = mode
        self.image_size = image_size
        self.augment = augment
        # augmentation_params: dict from config["data"]["augmentation"]
        self.augmentation_params = None

        # Collect image paths and labels
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        self._load_data()
        self.transform = self._get_transform()

    def _load_data(self):
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        class_names = sorted([d for d in os.listdir(self.data_dir)
                             if os.path.isdir(os.path.join(self.data_dir, d))])

        # Collect all samples
        all_samples = []
        for idx, class_name in enumerate(class_names):
            self.class_to_idx[class_name] = idx
            class_dir = os.path.join(self.data_dir, class_name)

            class_samples = []
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    class_samples.append((img_path, idx))

            # Split samples for this class (70% train, 30% val)
            random.shuffle(class_samples)
            n_train = int(len(class_samples) * 0.7)
            train_samples = class_samples[:n_train]
            val_samples = class_samples[n_train:]

            all_samples.extend(train_samples if self.mode == 'train' else val_samples)

        # Set the samples for this dataset
        self.image_paths, self.labels = zip(*all_samples) if all_samples else ([], [])

    def _get_transform(self):
        """Get appropriate transforms for face images."""
        # Allow optional augmentation parameters (random_resized_crop, random_erasing_prob)
        aug_params = self.augmentation_params or {}

        if self.mode == 'train' and self.augment:
            transform_list = []

            # Use RandomResizedCrop if configured, otherwise simple Resize
            if aug_params.get("random_resized_crop", False):
                transform_list.append(transforms.RandomResizedCrop(self.image_size))
            else:
                transform_list.append(transforms.Resize((self.image_size, self.image_size)))

            # Standard augmentations
            if aug_params.get("random_horizontal_flip", True):
                transform_list.append(transforms.RandomHorizontalFlip())
            if aug_params.get("random_rotation", 0):
                transform_list.append(transforms.RandomRotation(aug_params.get("random_rotation", 10)))
            if aug_params.get("color_jitter", False):
                transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))

            transform_list.append(transforms.ToTensor())

            # RandomErasing works on tensors; apply before normalization
            re_prob = float(aug_params.get("random_erasing_prob", 0.0) or 0.0)
            if re_prob > 0:
                transform_list.append(transforms.RandomErasing(p=re_prob))

            transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

            return transforms.Compose(transform_list)
        else:
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
            # Open image
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Return a blank image if loading fails
            image = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'path': img_path
        }

