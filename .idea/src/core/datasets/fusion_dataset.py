import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms


class FusionDataset(Dataset):
    """Fusion dataset that loads both face and fingerprint data.

    For now, this is a placeholder that can work with face data only.
    When fingerprint data becomes available, it should load both modalities.
    """

    def __init__(self, face_data_dir, fingerprint_data_dir=None, mode='train',
                 face_image_size=224, fingerprint_image_size=224,
                 transform=None):
        """
        Args:
            face_data_dir: Path to face data directory
            fingerprint_data_dir: Path to fingerprint data (optional, placeholder)
            mode: 'train' or 'val'
            face_image_size: Size for face images
            fingerprint_image_size: Size for fingerprint images
            transform: Optional custom transform
        """
        self.face_data_dir = face_data_dir
        self.fingerprint_data_dir = fingerprint_data_dir
        self.mode = mode
        self.face_image_size = face_image_size
        self.fingerprint_image_size = fingerprint_image_size

        # For now, we only have face data, so use face dataset as proxy
        # TODO: When fingerprint data is available, load paired face+fingerprint samples
        self.face_dataset = FaceDatasetProxy(face_data_dir, mode, face_image_size)

        # Placeholder for fingerprint data
        self.fingerprint_dataset = None
        if fingerprint_data_dir and os.path.exists(fingerprint_data_dir):
            # TODO: Implement fingerprint dataset loading
            pass

        self.class_to_idx = self.face_dataset.class_to_idx

    def __len__(self):
        return len(self.face_dataset)

    def __getitem__(self, idx):
        """Return fused sample with both modalities.

        For now, returns face data twice as placeholder for both modalities.
        TODO: Load actual paired face+fingerprint data.
        """
        face_sample = self.face_dataset[idx]

        # Placeholder: use face data as fingerprint data for now
        # In real implementation, this should load actual fingerprint data
        fingerprint_sample = face_sample.copy()  # Placeholder

        return {
            'face_image': face_sample['image'],
            'fingerprint_image': fingerprint_sample['image'],  # Placeholder
            'label': face_sample['label'],
            'face_path': face_sample.get('path'),
            'fingerprint_path': fingerprint_sample.get('path')  # Placeholder
        }


class FaceDatasetProxy:
    """Simple proxy to reuse face dataset logic for fusion placeholder."""

    def __init__(self, data_dir, mode, image_size):
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