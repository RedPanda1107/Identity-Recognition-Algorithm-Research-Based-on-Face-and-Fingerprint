import logging
import os
import torch
import random
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

_logger = logging.getLogger(__name__)


class FingerprintDataset(Dataset):
    """Fingerprint dataset with person-wise train/val/test split and Gallery/Query support.

    Label definition: Label = PersonID（只按人分类，不按手指）
    Examples:
      - 001_left_index_1.bmp -> person_id = "001" -> class 0
      - 001_left_middle_2.bmp -> person_id = "001" -> class 0  (same person!)
      - 001_right_thumb_1.bmp -> person_id = "001" -> class 0  (same person!)
      - 002_left_index_1.bmp -> person_id = "002" -> class 1  (different person!)

    Total classes: Number of unique persons (e.g., 200 persons -> 200 classes)

    Data split is PERSON-WISE:
        - train: 80% of persons
        - val:   10% of persons (50% of the remaining 20%)
        - test:  10% of persons (50% of the remaining 20%)
      Each person's ALL fingers go to exactly one split (no overlap).
      This ensures the model is tested on UNSEEN persons (true open-set scenario).
      Equivalent to FaceDataset's validation protocol.

    Gallery/Query split:
        - Gallery: up to N images per person (default 3)
        - Query: remaining images (excluding gallery)
      Both val and test sets have their own Gallery/Query splits.
    """

    def __init__(self, data_dir, mode='train', image_size=224, augment=True,
                 max_persons=None, class_to_idx=None, use_clahe=True,
                 split_ratio=0.8, test_split_ratio=0.5,
                 gallery_per_person=3, seed=42):
        """
        Args:
            data_dir: Path to fingerprint data directory
            mode: 'train', 'val', or 'test'
            image_size: Image size for resizing
            augment: Whether to apply data augmentation
            max_persons: Limit number of persons (for quick experiments)
            class_to_idx: Optional dict mapping person_id -> class index.
                           When provided, ensures consistent label space across train/val/test.
            use_clahe: Whether to apply CLAHE preprocessing for enhanced ridge contrast
            split_ratio: Train/val split ratio by person (default 0.8)
            test_split_ratio: Fraction of val persons for 'val' mode (default 0.5).
                              When 1.0, only val set exists (no test set).
            gallery_per_person: Number of gallery images per person (default 3)
            seed: Random seed for reproducible splits (default 42)
        """
        self.data_dir = data_dir
        self.mode = mode
        self.image_size = image_size
        self.augment = augment
        self.max_persons = max_persons
        self.use_clahe = use_clahe
        # Store CLAHE parameters (not the object itself, to avoid pickle issues on Windows)
        # Each worker process will create its own CLAHE via _get_clahe()
        self._clahe_params = (2.0, (8, 8)) if use_clahe else None
        self.split_ratio = split_ratio
        self.test_split_ratio = test_split_ratio
        self.gallery_per_person = gallery_per_person
        self.seed = seed

        self._provided_class_to_idx = class_to_idx
        self.augmentation_params = None

        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        self._load_data()
        self.transform = self._get_transform()

    def _load_data(self):
        """Load and prepare fingerprint dataset with person-wise train/val/test split."""
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        # Get all person directories (001, 002, 003, ...)
        person_dirs = sorted([d for d in os.listdir(self.data_dir)
                             if os.path.isdir(os.path.join(self.data_dir, d))])

        if self.max_persons and len(person_dirs) > self.max_persons:
            person_dirs = person_dirs[:self.max_persons]

        # Collect all images grouped by person_id
        person_samples = {}
        for person_id in person_dirs:
            person_dir = os.path.join(self.data_dir, person_id)

            for hand in ['L', 'R', 'left', 'right']:
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

        # ── Person-level label mapping ───────────────────────────────────────
        if self._provided_class_to_idx is not None:
            self.person_id_to_idx = self._provided_class_to_idx
            self.class_to_idx = self._provided_class_to_idx
            person_samples = {
                pid: paths for pid, paths in person_samples.items()
                if pid in self.person_id_to_idx
            }
        else:
            sorted_person_ids = sorted(person_samples.keys())
            self.person_id_to_idx = {pid: idx for idx, pid in enumerate(sorted_person_ids)}
            self.class_to_idx = self.person_id_to_idx

        # ── Person-wise split: train / val / test ─────────────────────────────
        random.seed(self.seed)
        all_person_ids = list(person_samples.keys())
        random.shuffle(all_person_ids)

        split_idx = int(len(all_person_ids) * self.split_ratio)
        train_person_ids = all_person_ids[:split_idx]
        val_person_ids = all_person_ids[split_idx:]

        # Val/Test split within val_person_ids
        if self.test_split_ratio < 1.0 and len(val_person_ids) >= 2:
            random.seed(self.seed + 100)
            val_person_list = list(val_person_ids)
            random.shuffle(val_person_list)

            n_val_persons = max(1, int(len(val_person_list) * self.test_split_ratio))
            n_val_persons = min(n_val_persons, len(val_person_list) - 1)

            val_only_person_ids = val_person_list[:n_val_persons]
            test_person_ids = val_person_list[n_val_persons:]
        else:
            val_only_person_ids = val_person_ids
            test_person_ids = []

        train_person_set = set(train_person_ids)
        val_person_set = set(val_only_person_ids)
        test_person_set = set(test_person_ids)

        train_samples = []
        val_samples = []
        test_samples = []

        for person_id, img_paths in person_samples.items():
            person_idx = self.person_id_to_idx[person_id]
            for img_path in img_paths:
                sample = {
                    'person_id': person_id,
                    'img_path': img_path,
                    'label_idx': person_idx
                }
                if person_id in train_person_set:
                    train_samples.append(sample)
                elif person_id in val_person_set:
                    val_samples.append(sample)
                elif person_id in test_person_set:
                    test_samples.append(sample)

        # ── Data integrity checks ────────────────────────────────────────────
        train_person_check = set(s['person_id'] for s in train_samples)
        val_person_check = set(s['person_id'] for s in val_samples)
        test_person_check = set(s['person_id'] for s in test_samples)

        train_val_overlap = train_person_check & val_person_check
        train_test_overlap = train_person_check & test_person_check
        val_test_overlap = val_person_check & test_person_check

        if train_val_overlap or train_test_overlap or val_test_overlap:
            _logger.error(
                f"[ERROR] 数据泄露! train∩val={train_val_overlap}, "
                f"train∩test={train_test_overlap}, val∩test={val_test_overlap}"
            )
        else:
            _logger.info(
                f"[PASS] Person-wise split: 训练集 {len(train_person_check)} 人, "
                f"验证集 {len(val_person_check)} 人, 测试集 {len(test_person_check)} 人，"
                f"人员完全不重叠"
            )

        train_paths_set = set(s['img_path'] for s in train_samples)
        val_paths_set = set(s['img_path'] for s in val_samples)
        test_paths_set = set(s['img_path'] for s in test_samples)

        train_val_path_overlap = train_paths_set & val_paths_set
        train_test_path_overlap = train_paths_set & test_paths_set
        val_test_path_overlap = val_paths_set & test_paths_set

        if train_val_path_overlap or train_test_path_overlap or val_test_path_overlap:
            _logger.error(
                f"[ERROR] 数据泄露! 图片路径重叠: "
                f"train∩val={len(train_val_path_overlap)}, "
                f"train∩test={len(train_test_path_overlap)}, "
                f"val∩test={len(val_test_path_overlap)}"
            )
        else:
            _logger.info("[PASS] 无数据泄露，训练/验证/测试集图片完全不重叠")

        _logger.info(f"训练集: {len(train_samples)} 张图 / {len(train_person_check)} 人")
        _logger.info(f"验证集: {len(val_samples)} 张图 / {len(val_person_check)} 人")
        if test_person_check:
            _logger.info(f"测试集: {len(test_samples)} 张图 / {len(test_person_check)} 人")

        # ── Gallery / Query split for val and test ───────────────────────────
        def _split_gallery_query(person_ids_list, samples_list, prefix=""):
            """Helper: split samples into gallery and query per person."""
            random.seed(self.seed + 200)
            gallery_paths, gallery_labels = [], []
            query_paths, query_labels = [], []

            for person_id in person_ids_list:
                person_idx = self.person_id_to_idx[person_id]
                person_paths = [s['img_path'] for s in samples_list if s['person_id'] == person_id]

                # Random shuffle before split (avoid fixed-order bias)
                shuffled_paths = person_paths.copy()
                random.shuffle(shuffled_paths)

                gallery_paths_p = shuffled_paths[:self.gallery_per_person]
                query_paths_p = shuffled_paths[self.gallery_per_person:]

                gallery_paths.extend(gallery_paths_p)
                gallery_labels.extend([person_idx] * len(gallery_paths_p))
                query_paths.extend(query_paths_p)
                query_labels.extend([person_idx] * len(query_paths_p))

            if prefix:
                _logger.info(f"{prefix} Gallery: {len(gallery_paths)} 张（每人 × {self.gallery_per_person} 张）")
                _logger.info(f"{prefix} Query:   {len(query_paths)} 张（每人去除 gallery 后的图）")

            return gallery_paths, gallery_labels, query_paths, query_labels

        # Val Gallery/Query
        val_person_ids_list = list(val_person_set)
        (self.val_gallery_paths, self.val_gallery_labels,
         self.val_query_paths, self.val_query_labels) = \
            _split_gallery_query(val_person_ids_list, val_samples, prefix="Val")

        # Test Gallery/Query
        if test_person_set:
            test_person_ids_list = list(test_person_set)
            (self.test_gallery_paths, self.test_gallery_labels,
             self.test_query_paths, self.test_query_labels) = \
                _split_gallery_query(test_person_ids_list, test_samples, prefix="Test")
        else:
            self.test_gallery_paths = []
            self.test_gallery_labels = []
            self.test_query_paths = []
            self.test_query_labels = []

        # ── Select samples for current mode ─────────────────────────────────
        if self.mode == 'train':
            filtered_samples = train_samples
        elif self.mode == 'val':
            filtered_samples = val_samples
        elif self.mode == 'test':
            filtered_samples = test_samples
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Must be 'train', 'val', or 'test'.")

        self.image_paths = [s['img_path'] for s in filtered_samples]
        self.labels = [s['label_idx'] for s in filtered_samples]

        _logger.info(
            f"FingerprintDataset mode={self.mode}, "
            f"samples={len(self.image_paths)}, "
            f"classes={len(self.class_to_idx)}"
        )

    def _get_transform(self):
        """Get appropriate transforms for fingerprint images with full augmentation support."""
        aug = self.augmentation_params or {}

        if self.mode == 'train' and self.augment:
            t = []

            # 旋转（模拟按压偏位）
            rot = aug.get("random_rotation", 0)
            if rot:
                t.append(transforms.RandomRotation(rot))

            # 轻微平移 + 缩放（模拟按压力度/位置差异）
            translate = aug.get("translate", [0.0, 0.0])
            scale = aug.get("scale", [1.0, 1.0])
            if translate != [0.0, 0.0] or scale != [1.0, 1.0]:
                t.append(transforms.RandomAffine(
                    degrees=0, translate=translate, scale=scale
                ))

            t.append(transforms.Resize((self.image_size, self.image_size)))

            # 灰度对比度/亮度波动（指纹采集差异）
            if aug.get("color_jitter", False):
                brightness = aug.get("color_jitter_brightness", 0.2)
                contrast = aug.get("color_jitter_contrast", 0.2)
                saturation = aug.get("color_jitter_saturation", 0.0)
                hue = aug.get("color_jitter_hue", 0.0)
                t.append(transforms.ColorJitter(
                    brightness=brightness, contrast=contrast,
                    saturation=saturation, hue=hue
                ))

            t.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # 随机遮挡（模拟指纹缺损/污渍）
            re_prob = aug.get("random_erasing_prob", 0.0)
            if re_prob > 0:
                re_scale = tuple(aug.get("random_erasing_scale", [0.02, 0.15]))
                t.append(transforms.RandomErasing(p=re_prob, scale=re_scale))

            return transforms.Compose(t)

        # 验证/推理：固定处理，无随机性
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def _get_clahe(self):
        """Lazily create CLAHE object per worker (cached after first call)."""
        if not self.use_clahe:
            return None
        if not hasattr(self, '_clahe_cached'):
            self._clahe_cached = cv2.createCLAHE(
                clipLimit=self._clahe_params[0],
                tileGridSize=self._clahe_params[1]
            )
        return self._clahe_cached

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('L')
            image = Image.merge('RGB', [image, image, image])

            # CLAHE 预处理：增强指纹脊线对比度（每个 worker 独立创建，避免 pickle）
            if self.use_clahe:
                img_array = np.array(image)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
                clahe = self._get_clahe()
                enhanced = clahe.apply(gray)
                image = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))
        except Exception:
            image = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'path': img_path
        }
