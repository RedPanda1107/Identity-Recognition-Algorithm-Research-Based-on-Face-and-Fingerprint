import os
import logging
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random
import numpy as np

_logger = logging.getLogger(__name__)


class FaceDataset(Dataset):
    """Face dataset with person-wise split and open-set evaluation support.

    三分割支持（可选）：
        - 训练集（train）：Person-wise split，80% 的人用于训练
        - 验证集（val）：与训练集完全不重叠的人中，test_split_ratio 比例用于早停
        - 测试集（test）：验证集中剩余的人，仅用于最终评估
        - Val/Test 内部 Gallery/Query split：每人前 3 张 → Gallery，剩余 → Query
        - 验证方式：1:N 余弦相似度检索（Rank-K + EER）

    二分割（默认）：
        - 训练集：80% 的人
        - 验证集：20% 的人（同时用于早停和最终评估）

    数据目录结构（与 FingerprintDataset 保持一致）：
        data_dir/
            person_001/
                img1.jpg
                img2.jpg
                ...
            person_002/
                ...
    """

    def __init__(self, data_dir, mode='train', image_size=224, augment=True,
                 max_persons=None, class_to_idx=None, gallery_per_person=3,
                 val_split_ratio=0.8, test_split_ratio=0.5, seed=42):
        """
        Args:
            data_dir: Path to face data directory (按 person 划分子目录)
            mode: 'train', 'val', 或 'test'
            image_size: Image size for resizing
            augment: Whether to apply data augmentation
            max_persons: Limit number of persons (for quick experiments)
            class_to_idx: Optional dict mapping person_id -> class index.
                           When provided, ensures consistent label space across train/val.
            gallery_per_person: Number of images per person in val/test Gallery (default 3)
            val_split_ratio: Train/val split ratio by person (default 0.8)
            test_split_ratio: Fraction of val persons for 'val' mode, rest for 'test' mode (default 0.5)
                              When 1.0, only val set exists (no test set).
            seed: Random seed for reproducible splits
        """
        self.data_dir = data_dir
        self.mode = mode
        self.image_size = image_size
        self.augment = augment
        self.max_persons = max_persons
        self.gallery_per_person = gallery_per_person
        self.split_ratio = val_split_ratio
        self.test_split_ratio = test_split_ratio  # 用于在 val 中划分 val/test 人员
        self.seed = seed

        self._provided_class_to_idx = class_to_idx
        self.augmentation_params = None

        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        self._load_data()
        self.transform = self._get_transform()

    def _load_data(self):
        """Load and prepare face dataset with person-wise split (supports train/val/test split)."""
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        person_dirs = sorted([d for d in os.listdir(self.data_dir)
                             if os.path.isdir(os.path.join(self.data_dir, d))])

        if self.max_persons and len(person_dirs) > self.max_persons:
            person_dirs = person_dirs[:self.max_persons]

        # Group all images by person_id
        person_samples = {}
        for person_id in person_dirs:
            person_dir = os.path.join(self.data_dir, person_id)
            img_paths = sorted([
                os.path.join(person_dir, f)
                for f in os.listdir(person_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ])
            if img_paths:
                person_samples[person_id] = img_paths

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

        # ── Person-wise split ───────────────────────────────────────────────
        random.seed(self.seed)
        all_person_ids = list(person_samples.keys())
        random.shuffle(all_person_ids)

        split_idx = int(len(all_person_ids) * self.split_ratio)
        train_person_ids = all_person_ids[:split_idx]
        val_person_ids = all_person_ids[split_idx:]
        train_person_set = set(train_person_ids)
        val_person_set = set(val_person_ids)

        # ── 验证集内部分割: val_person_ids → val_person_ids + test_person_ids ─
        # test_split_ratio: val 集合占验证集人员的比例 (默认 0.5, 即 50% val, 50% test)
        if self.test_split_ratio < 1.0 and len(val_person_ids) >= 2:
            # 使用相同种子确保可复现
            random.seed(self.seed + 100)  # 与训练划分使用不同种子避免干扰
            val_person_list = list(val_person_ids)
            random.shuffle(val_person_list)

            n_val_persons = max(1, int(len(val_person_list) * self.test_split_ratio))
            # 确保每个 split 至少有 1 个人
            n_val_persons = min(n_val_persons, len(val_person_list) - 1)

            val_only_person_ids = val_person_list[:n_val_persons]
            test_person_ids = val_person_list[n_val_persons:]
            val_person_set = set(val_only_person_ids)
            test_person_set = set(test_person_ids)
        else:
            # test_split_ratio = 1.0 时，不分割，所有验证人员都用于 val
            val_only_person_ids = val_person_ids
            test_person_ids = []
            val_person_set = set(val_only_person_ids)
            test_person_set = set()

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
        train_person_set_check = set(s['person_id'] for s in train_samples)
        val_person_set_check = set(s['person_id'] for s in val_samples)
        test_person_set_check = set(s['person_id'] for s in test_samples)

        # 检查三集合之间的重叠
        train_val_overlap = train_person_set_check & val_person_set_check
        train_test_overlap = train_person_set_check & test_person_set_check
        val_test_overlap = val_person_set_check & test_person_set_check

        if train_val_overlap or train_test_overlap or val_test_overlap:
            _logger.error(f"[ERROR] 数据泄露! train∩val={train_val_overlap}, train∩test={train_test_overlap}, val∩test={val_test_overlap}")
        else:
            _logger.info(f"[PASS] Person-wise split: 训练集 {len(train_person_set_check)} 人, "
                         f"验证集 {len(val_person_set_check)} 人, 测试集 {len(test_person_set_check)} 人")
            _logger.info(f"[PASS] 人员完全不重叠")

        train_paths_set = set(s['img_path'] for s in train_samples)
        val_paths_set = set(s['img_path'] for s in val_samples)
        test_paths_set = set(s['img_path'] for s in test_samples)

        train_val_path_overlap = train_paths_set & val_paths_set
        train_test_path_overlap = train_paths_set & test_paths_set
        val_test_path_overlap = val_paths_set & test_paths_set

        if train_val_path_overlap or train_test_path_overlap or val_test_path_overlap:
            _logger.error(f"[ERROR] 数据泄露! 图片路径重叠: "
                          f"train∩val={len(train_val_path_overlap)}, "
                          f"train∩test={len(train_test_path_overlap)}, "
                          f"val∩test={len(val_test_path_overlap)}")
        else:
            _logger.info("[PASS] 无数据泄露，训练/验证/测试集图片完全不重叠")

        _logger.info(f"训练集: {len(train_samples)} 张图 / {len(train_person_set_check)} 人")
        _logger.info(f"验证集: {len(val_samples)} 张图 / {len(val_person_set_check)} 人")
        if test_person_set_check:
            _logger.info(f"测试集: {len(test_samples)} 张图 / {len(test_person_set_check)} 人")

        # ── Val/Test Gallery / Query split (person-wise, random) ─────────────
        # Gallery: randomly selected N images per person
        # Query: remaining images (excluding gallery)
        #
        # 随机划分避免固定划分带来的顺序偏差（如光照/角度渐变导致的系统性偏差）

        def _split_gallery_query(person_ids_list, samples_list):
            """Helper: 对给定的人员列表随机划分 gallery 和 query"""
            random.seed(self.seed + 200)  # 与数据划分种子不同，确保随机性
            gallery_paths, gallery_labels, gallery_person_ids = [], [], []
            query_paths, query_labels, query_person_ids = [], [], []

            for person_id in person_ids_list:
                person_idx = self.person_id_to_idx[person_id]
                person_paths = [s['img_path'] for s in samples_list if s['person_id'] == person_id]

                # 随机打乱图片顺序后再划分（避免固定顺序偏差）
                shuffled_paths = person_paths.copy()
                random.shuffle(shuffled_paths)

                gallery_paths_p = shuffled_paths[:self.gallery_per_person]
                query_paths_p = shuffled_paths[self.gallery_per_person:]

                for g_path in gallery_paths_p:
                    gallery_paths.append(g_path)
                    gallery_labels.append(person_idx)
                    gallery_person_ids.append(person_id)

                for q_path in query_paths_p:
                    query_paths.append(q_path)
                    query_labels.append(person_idx)
                    query_person_ids.append(person_id)

            return (gallery_paths, gallery_labels, gallery_person_ids,
                    query_paths, query_labels, query_person_ids)

        # 验证集 gallery/query 划分
        val_person_ids_list = list(val_person_set)
        (self.val_gallery_paths, self.val_gallery_labels, self.val_gallery_person_ids,
         self.val_query_paths, self.val_query_labels, self.val_query_person_ids) = \
            _split_gallery_query(val_person_ids_list, val_samples)

        _logger.info(f"Val Gallery: {len(self.val_gallery_paths)} 张（验证人 × {self.gallery_per_person} 张）")
        _logger.info(f"Val Query:   {len(self.val_query_paths)} 张（验证人去除 gallery 后的图）")

        # 测试集 gallery/query 划分（如果有测试集）
        if test_person_set_check:
            test_person_ids_list = list(test_person_set)
            (self.test_gallery_paths, self.test_gallery_labels, self.test_gallery_person_ids,
             self.test_query_paths, self.test_query_labels, self.test_query_person_ids) = \
                _split_gallery_query(test_person_ids_list, test_samples)

            _logger.info(f"Test Gallery: {len(self.test_gallery_paths)} 张（测试人 × {self.gallery_per_person} 张）")
            _logger.info(f"Test Query:   {len(self.test_query_paths)} 张（测试人去除 gallery 后的图）")
        else:
            self.test_gallery_paths = []
            self.test_gallery_labels = []
            self.test_gallery_person_ids = []
            self.test_query_paths = []
            self.test_query_labels = []
            self.test_query_person_ids = []

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

        _logger.info(f"FaceDataset mode={self.mode}, "
                     f"samples={len(self.image_paths)}, "
                     f"classes={len(self.class_to_idx)}")

    def _get_transform(self):
        """Get appropriate transforms for face images."""
        aug_params = self.augmentation_params or {}

        if self.mode == 'train' and self.augment:
            transform_list = []

            if aug_params.get("random_resized_crop", False):
                transform_list.append(transforms.RandomResizedCrop(self.image_size))
            else:
                transform_list.append(transforms.Resize((self.image_size, self.image_size)))

            if aug_params.get("random_horizontal_flip", True):
                transform_list.append(transforms.RandomHorizontalFlip())
            if aug_params.get("random_rotation", 0):
                transform_list.append(transforms.RandomRotation(aug_params.get("random_rotation", 10)))
            if aug_params.get("color_jitter", False):
                transform_list.append(transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ))

            transform_list.append(transforms.ToTensor())

            re_prob = float(aug_params.get("random_erasing_prob", 0.0) or 0.0)
            if re_prob > 0:
                transform_list.append(transforms.RandomErasing(p=re_prob))

            transform_list.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ))
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
