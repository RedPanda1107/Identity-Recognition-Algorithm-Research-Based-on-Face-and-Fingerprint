import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random


class FingerprintDataset(Dataset):
    """Fingerprint dataset implementation for identification tasks."""

    def __init__(self, data_dir, mode='train', image_size=224, augment=True):
        """
        Args:
            data_dir: Path to fingerprint data directory
            mode: 'train' or 'val'
            image_size: Image size for resizing
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.mode = mode
        self.image_size = image_size
        self.augment = augment

        # Collect image paths and labels
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        self._load_data()
        self.transform = self._get_transform()

    def _load_data(self):
        """Load fingerprint data organized by person ID."""
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        # Get all person directories (001, 002, 003, ...)
        person_dirs = sorted([d for d in os.listdir(self.data_dir)
                             if os.path.isdir(os.path.join(self.data_dir, d))])

        for idx, person_id in enumerate(person_dirs):
            self.class_to_idx[person_id] = idx
            person_dir = os.path.join(self.data_dir, person_id)

            # Collect images from both left and right hand directories
            for hand in ['left', 'right']:
                hand_dir = os.path.join(person_dir, hand)
                if not os.path.exists(hand_dir):
                    continue

                for finger_file in os.listdir(hand_dir):
                    if finger_file.lower().endswith('.bmp'):
                        img_path = os.path.join(hand_dir, finger_file)
                        self.image_paths.append(img_path)
                        self.labels.append(idx)

    def _get_transform(self):
        """Get appropriate transforms for fingerprint images."""
        if self.mode == 'train' and self.augment:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomRotation(15),  # Fingerprints can have slight rotation variations
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Elastic deformation simulation
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Handle varying scan conditions
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
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
            # Open BMP image
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