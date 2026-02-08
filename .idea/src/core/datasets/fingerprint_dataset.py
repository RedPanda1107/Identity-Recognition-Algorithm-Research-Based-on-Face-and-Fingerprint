import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


# CLAHE 预处理器 - 指纹对比度增强
class CLAHEPreprocessor:
    """CLAHE (对比度受限的直方图均衡化) 预处理器，用于指纹图像增强."""

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Args:
            clip_limit: 对比度限制值
            tile_grid_size: 网格大小
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def apply(self, image):
        """
        对图像应用 CLAHE 处理.

        Args:
            image: PIL RGB 图像或 numpy array

        Returns:
            处理后的图像 (numpy array, RGB格式)
        """
        # 转为 numpy array (如果输入是 PIL)
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        # 确保是 RGB 格式 (3通道)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 1:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif img_array.shape[2] == 3:
            img_array = img_array.copy()  # 已经是 RGB

        # 转灰度图进行 CLAHE 处理
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # 应用 CLAHE
        clahe_img = self.clahe.apply(gray)

        # 转回 RGB
        result = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)

        return result

    def __call__(self, image):
        """可调用接口."""
        return self.apply(image)


# 使用 INTER_AREA 的 Resize - 保留脊线细节
class ResizeWithInterArea:
    """使用 cv2.INTER_AREA 的图像缩放，用于保留指纹脊线细节."""

    def __init__(self, size):
        """
        Args:
            size: 目标大小 (int 或 tuple)
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)

    def __call__(self, img):
        """
        对 PIL 图像进行缩放.

        Args:
            img: PIL Image

        Returns:
            缩放后的 PIL Image
        """
        # PIL 转 numpy
        img_array = np.array(img)

        # 使用 INTER_AREA 进行缩放
        resized = cv2.resize(img_array, self.size[::-1], interpolation=cv2.INTER_AREA)

        # 转回 PIL
        return Image.fromarray(resized)

    def __repr__(self):
        return f"ResizeWithInterArea(size={self.size})"


class FingerprintDataset(Dataset):
    """Fingerprint dataset implementation for finger identification tasks.

    Label definition: Label = PersonID + Hand + Finger (unique finger ID)
    Examples:
      - 001_left_index   -> class 0
      - 001_left_thumb   -> class 1
      - 001_right_index  -> class 2
      - 001_right_thumb  -> class 3
      - 002_left_index   -> class 4
      ...

    With 600 persons, 2 hands, 5 fingers each -> num_classes = 600 × 10 = 6000

    Each finger has 4 image variants. Data split is RANDOM by sample (NOT by person):
    - For each finger's 4 images: randomly 3 for training, 1 for validation
    - This ensures the model can evaluate on seen fingers during validation
    """

    def __init__(self, data_dir, mode='train', image_size=224, augment=True, max_persons=None):
        """
        Args:
            data_dir: Path to fingerprint data directory
            mode: 'train' or 'val'
            image_size: Image size for resizing
            augment: Whether to apply data augmentation
            max_persons: Limit number of persons (for quick experiments)
        """
        self.data_dir = data_dir
        self.mode = mode
        self.image_size = image_size
        self.augment = augment
        self.max_persons = max_persons

        # CLAHE 预处理器 - 指纹对比度增强
        self.clahe_preprocessor = CLAHEPreprocessor(
            clip_limit=2.0,
            tile_grid_size=(8, 8)
        )

        # optional augmentation parameters (from config["data"]["augmentation"])
        self.augmentation_params = None

        # Collect image paths and labels
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        self._load_data()
        self.transform = self._get_transform()

    def _load_data(self):
        """Load fingerprint data organized by PERSON ID for PERSON IDENTIFICATION.

        Label definition: Label = PersonID
        Each person has multiple fingers (left/right × 5 fingers).
        All fingers from the same person share the same label.

        Data split is BY PERSON:
        - 75% of persons for training
        - 25% of persons for validation
        This ensures the model learns to identify people, not just fingers.

        Examples:
          - 001_left_index_1.bmp -> person_id = "001", class = 0
          - 001_left_index_2.bmp -> person_id = "001", class = 0
          - 001_right_index_1.bmp -> person_id = "001", class = 0
          - 002_left_1.bmp -> person_id = "002", class = 1
          ...

        Total classes: Number of unique persons (e.g., 100 persons = 100 classes)
        """
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        # Get all person directories (001, 002, 003, ...)
        person_dirs = sorted([d for d in os.listdir(self.data_dir)
                             if os.path.isdir(os.path.join(self.data_dir, d))])

        # Limit to first 100 persons for quick experiments
        max_persons = getattr(self, 'max_persons', None)
        if max_persons and len(person_dirs) > max_persons:
            person_dirs = person_dirs[:max_persons]

        # Collect all samples grouped by person
        # person_id -> list of image paths
        person_samples = {}  # person_id: [img_path, ...]

        for person_id in person_dirs:
            person_dir = os.path.join(self.data_dir, person_id)

            for hand in ['left', 'right']:
                hand_dir = os.path.join(person_dir, hand)
                if not os.path.exists(hand_dir):
                    continue

                for finger_file in sorted(os.listdir(hand_dir)):
                    if not finger_file.lower().endswith('.bmp'):
                        continue

                    img_path = os.path.join(hand_dir, finger_file)

                    if person_id not in person_samples:
                        person_samples[person_id] = []
                    person_samples[person_id].append(img_path)

        # Create person-to-index mapping
        sorted_person_ids = sorted(person_samples.keys())
        self.person_id_to_idx = {person_id: idx for idx, person_id in enumerate(sorted_person_ids)}
        self.class_to_idx = self.person_id_to_idx

        # Split samples by PERSON (not by finger)
        # This ensures: all fingers from the same person go to the same split
        train_persons = []
        val_persons = []

        # Shuffle person IDs for random split
        shuffled_person_ids = list(sorted_person_ids)
        random.shuffle(shuffled_person_ids)

        # 75% train, 25% val (by person)
        n_train_persons = max(1, int(len(shuffled_person_ids) * 0.75))
        n_val_persons = len(shuffled_person_ids) - n_train_persons

        if n_val_persons == 0 and len(shuffled_person_ids) >= 2:
            n_train_persons = len(shuffled_person_ids) - 1
            n_val_persons = 1

        train_persons = shuffled_person_ids[:n_train_persons]
        val_persons = shuffled_person_ids[n_train_persons:]

        train_samples = []
        val_samples = []

        for person_id in train_persons:
            person_idx = self.person_id_to_idx[person_id]
            for img_path in person_samples[person_id]:
                train_samples.append({
                    'person_id': person_id,
                    'img_path': img_path,
                    'label_idx': person_idx
                })

        for person_id in val_persons:
            person_idx = self.person_id_to_idx[person_id]
            for img_path in person_samples[person_id]:
                val_samples.append({
                    'person_id': person_id,
                    'img_path': img_path,
                    'label_idx': person_idx
                })

        # Select samples based on mode
        if self.mode == 'train':
            filtered_samples = train_samples
        else:
            filtered_samples = val_samples

        # Extract image_paths and labels
        self.image_paths = [s['img_path'] for s in filtered_samples]
        self.labels = [s['label_idx'] for s in filtered_samples]

        print(f"[FingerprintDataset] Mode: {self.mode}, "
              f"Samples: {len(self.image_paths)}, "
              f"Unique Persons (Classes): {len(self.class_to_idx)}")

    def _get_transform(self):
        """Get appropriate transforms for fingerprint images.

        Optimized augmentation strategy for fingerprints:
        - Remove RandomResizedCrop: destroys ridge frequency patterns
        - Keep ResizeWithInterArea: preserves ridge details
        - Keep small RandomRotation (<= 10 degrees): minimal geometric distortion
        """
        aug_params = self.augmentation_params or {}
        transform_list = []

        if self.mode == 'train' and self.augment:
            # 使用 INTER_AREA Resize 保留指纹脊线细节
            # 移除 RandomResizedCrop：它会破坏指纹的脊线频率
            transform_list.append(ResizeWithInterArea(self.image_size))

            # 微小的随机旋转（限制在 10 度以内）
            # 避免过大旋转导致指纹方向完全改变
            rot_degrees = min(aug_params.get("random_rotation", 10), 10)
            if rot_degrees > 0:
                transform_list.append(transforms.RandomRotation(degrees=rot_degrees))

            # 指纹禁用 RandomAffine，因为纹理对几何变换敏感
            # 即使启用也只用极小参数
            if aug_params.get("random_affine", False):
                transform_list.append(transforms.RandomAffine(
                    degrees=0,
                    translate=(0.02, 0.02),
                    scale=(0.99, 1.01)
                ))

            # 轻微的颜色抖动
            if aug_params.get("color_jitter", True):
                transform_list.append(transforms.ColorJitter(
                    brightness=0.05,
                    contrast=0.05
                ))

            transform_list.append(transforms.ToTensor())

            # RandomErasing support (applied on tensor)
            re_prob = float(aug_params.get("random_erasing_prob", 0.0) or 0.0)
            if re_prob > 0:
                transform_list.append(transforms.RandomErasing(p=re_prob))

            transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        else:
            transform_list = [
                ResizeWithInterArea(self.image_size),  # 使用 INTER_AREA 保留脊线细节
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]

        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # Open BMP image
            image = Image.open(img_path).convert('RGB')

            # 应用 CLAHE 对比度增强 (使用预创建的 CLAHE 避免重复创建)
            # CLAHEPreprocessor.apply() 已经完成：RGB -> 灰度 -> CLAHE -> RGB
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            enhanced_rgb = self.clahe_preprocessor.apply(gray)  # 直接返回 RGB 图像
            image = Image.fromarray(enhanced_rgb)

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
