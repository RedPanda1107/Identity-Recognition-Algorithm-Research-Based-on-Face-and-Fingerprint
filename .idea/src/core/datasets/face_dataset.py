import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class FaceDataset(Dataset):
    """Face dataset implementation."""

    def __init__(self, data_dir, mode='train', image_size=224):
        """
        Args:
            data_dir: Path to face data directory
            mode: 'train' or 'val'
            image_size: Image size for resizing
        """
        self.data_dir = data_dir
        self.mode = mode
        self.image_size = image_size

        # Collect image paths and labels
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        self._load_data()
        self.transform = self._get_default_transform()

    def _load_data(self):
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        class_names = sorted([d for d in os.listdir(self.data_dir)
                             if os.path.isdir(os.path.join(self.data_dir, d))])

        for idx, class_name in enumerate(class_names):
            self.class_to_idx[class_name] = idx
            class_dir = os.path.join(self.data_dir, class_name)

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(idx)

    def _get_default_transform(self):
        if self.mode == 'train':
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
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
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'path': img_path
        }

