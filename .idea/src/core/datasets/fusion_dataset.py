import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
import pandas as pd
import random
import logging
import cv2
import numpy as np

_logger = logging.getLogger(__name__)

# 预定义的归一化参数（集中管理，与所有数据集和推理 pipeline 统一）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)


class FusionDataset(Dataset):
    """多模态人脸+指纹融合数据集

    支持真正的人脸-指纹配对数据加载
    支持 person-wise train/val/test 三层划分
    """

    def __init__(self, face_data_dir, fingerprint_data_dir,
                 mapping_file=None, mode='train',
                 face_image_size=224, fingerprint_image_size=224,
                 transform=None, augment=True, seed=42,
                 split_ratio=0.8, test_split_ratio=0.5,
                 gallery_per_person=3, class_to_idx=None,
                 use_clahe=True):
        """
        Args:
            face_data_dir: 人脸数据目录路径
            fingerprint_data_dir: 指纹数据目录路径
            mapping_file: 人脸-指纹映射文件路径 (CSV或JSON)
            mode: 'train', 'val', 或 'test'
            face_image_size: 人脸图像尺寸
            fingerprint_image_size: 指纹图像尺寸
            transform: 自定义变换
            augment: 是否使用数据增强
            seed: 随机种子，用于确保 train/val/test 划分一致性
            split_ratio: Person-wise train/val split ratio (default 0.8)
            test_split_ratio: 在剩余人员中，val/test 划分比例 (default 0.5)
            gallery_per_person: Number of gallery samples per person (default 3)
            class_to_idx: 共享的类别映射（从训练集传入）
            use_clahe: 是否对指纹图像应用 CLAHE 增强（增强脊线对比度，默认 True）
        """
        self.face_data_dir = face_data_dir
        self.fingerprint_data_dir = fingerprint_data_dir
        self.data_dir = os.path.dirname(os.path.dirname(face_data_dir))
        self.mapping_file = mapping_file
        self.mode = mode
        self.face_image_size = face_image_size
        self.fingerprint_image_size = fingerprint_image_size
        self.augment = augment
        self.seed = seed
        self.split_ratio = split_ratio
        self.test_split_ratio = test_split_ratio
        self.gallery_per_person = gallery_per_person
        self.augmentation_params = None
        self._external_class_to_idx = class_to_idx
        self.use_clahe = use_clahe

        # 加载人脸-指纹映射
        if mapping_file:
            self.face_fp_mapping = self._load_mapping(mapping_file)
        else:
            self.face_fp_mapping = self._create_default_mapping()

        # 收集样本
        self.samples = []
        self.class_to_idx = {}
        self._collect_samples()

        # 设置变换
        self.transform = transform if transform else self._get_default_transform()

    def _load_mapping(self, mapping_file):
        """加载人脸-指纹映射文件"""
        if mapping_file.endswith('.csv'):
            df = pd.read_csv(mapping_file)
            mapping = {}
            for _, row in df.iterrows():
                face_id = str(row['face_id'])
                fp_id = str(row['fingerprint_id'])
                label = int(row['class_label'])
                mapping[face_id] = {'fingerprint_id': fp_id, 'label': label}
            return mapping
        elif mapping_file.endswith('.json'):
            import json
            with open(mapping_file, 'r') as f:
                data = json.load(f)
                mapping = {}
                for face_id, info in data.items():
                    person_id = info['person_id']
                    fp_id = info['fingerprint_id']
                    label = int(person_id)
                    mapping[face_id] = {
                        'fingerprint_id': fp_id,
                        'label': label,
                        'person_id': person_id,
                        'face_images': info.get('face_images', []),
                        'fingerprint_images': info.get('fingerprint_images', {})
                    }
                return mapping
        else:
            raise ValueError(f"不支持的映射文件格式: {mapping_file}")

    def _create_default_mapping(self):
        """创建默认映射（基于目录名匹配）"""
        mapping = {}
        if os.path.exists(self.face_data_dir):
            face_dirs = [d for d in os.listdir(self.face_data_dir)
                        if os.path.isdir(os.path.join(self.face_data_dir, d))]
            for face_id in face_dirs:
                mapping[face_id] = {
                    'fingerprint_id': face_id,
                    'label': len(mapping)
                }
        return mapping

    def _collect_samples(self):
        """收集所有有效的样本，并进行 train/val/test 三层划分"""
        random.seed(self.seed)

        all_samples = []

        for face_id, info in self.face_fp_mapping.items():
            fp_id = info['fingerprint_id']
            label = info['label']

            # 从映射文件中获取图像路径
            face_image_paths_raw = info.get('face_images', [])
            fingerprint_images = info.get('fingerprint_images', {})

            # 转换人脸图像路径为绝对路径
            face_image_paths = []
            for img_path in face_image_paths_raw:
                abs_path = os.path.join(self.data_dir, img_path.replace('\\', os.sep))
                face_image_paths.append(abs_path)

            # 如果映射文件中没有图像路径，则使用目录扫描
            if not face_image_paths:
                face_dir = os.path.join(self.face_data_dir, face_id)
                if os.path.exists(face_dir):
                    for img_name in os.listdir(face_dir):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            face_image_paths.append(os.path.join(face_dir, img_name))

            # 收集指纹图像路径
            fp_image_paths = []
            if fingerprint_images:
                for hand, images in fingerprint_images.items():
                    if isinstance(images, list):
                        for img_path in images:
                            abs_path = os.path.join(self.data_dir, img_path.replace('\\', os.sep))
                            fp_image_paths.append(abs_path)
            else:
                fp_base_dir = os.path.join(self.fingerprint_data_dir, fp_id)
                if os.path.exists(fp_base_dir):
                    for hand in ['left', 'right']:
                        hand_dir = os.path.join(fp_base_dir, hand)
                        if os.path.exists(hand_dir):
                            for img_name in os.listdir(hand_dir):
                                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                    fp_image_paths.append(os.path.join(hand_dir, img_name))

            # 确定性配对：一一对应配对，不使用循环配对（避免随机配对导致标签混淆）
            if face_image_paths and fp_image_paths:
                num_pairs = min(len(face_image_paths), len(fp_image_paths))
                # 固定配对：第 i 个人脸配对第 i 个指纹
                for i in range(num_pairs):
                    all_samples.append({
                        'face_path': face_image_paths[i],
                        'fingerprint_path': fp_image_paths[i],
                        'face_id': face_id,
                        'fingerprint_id': fp_id,
                        'label': label
                    })

        # 按 person 分组样本
        from collections import defaultdict
        samples_by_person = defaultdict(list)
        for sample in all_samples:
            samples_by_person[sample['face_id']].append(sample)

        # ── Person-wise split: train / val / test ─────────────────────────────
        all_person_ids = list(samples_by_person.keys())
        random.seed(self.seed)
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

        for person_id, person_samples in samples_by_person.items():
            if person_id in train_person_set:
                train_samples.extend(person_samples)
            elif person_id in val_person_set:
                val_samples.extend(person_samples)
            elif person_id in test_person_set:
                test_samples.extend(person_samples)

        # ── 数据完整性检查 ───────────────────────────────────────────────────
        train_val_overlap = train_person_set & val_person_set
        train_test_overlap = train_person_set & test_person_set
        val_test_overlap = val_person_set & test_person_set

        if train_val_overlap or train_test_overlap or val_test_overlap:
            _logger.error(
                f"[ERROR] 数据泄露! train∩val={train_val_overlap}, "
                f"train∩test={train_test_overlap}, val∩test={val_test_overlap}"
            )
        else:
            _logger.info(
                f"[PASS] Person-wise split: 训练集 {len(train_person_set)} 人, "
                f"验证集 {len(val_person_set)} 人, 测试集 {len(test_person_set)} 人，"
                f"人员完全不重叠"
            )

        # ── 类别映射（必须在 Gallery/Query split 之前创建）─────────────────────
        if self._external_class_to_idx is not None:
            self.class_to_idx = self._external_class_to_idx
        else:
            unique_labels = sorted(set(sample['label'] for sample in all_samples))
            self.class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        # ── 创建 Gallery/Query 划分 ──────────────────────────────────────────
        self._create_gallery_query_split(val_samples, 'val')
        self._create_gallery_query_split(test_samples, 'test')

        # ── 根据 mode 返回对应样本 ───────────────────────────────────────────
        if self.mode == 'train':
            self.samples = train_samples
        elif self.mode == 'val':
            self.samples = val_samples
        elif self.mode == 'test':
            self.samples = test_samples
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Must be 'train', 'val', or 'test'")

        # 更新样本标签为连续索引
        for sample in self.samples:
            sample['label'] = self.class_to_idx[sample['label']]

    def _create_gallery_query_split(self, samples, split_name='val'):
        """Create Gallery/Query split for val or test set (1:N retrieval).

        Uses random shuffle (seed = self.seed + 200) for reproducibility.
        This ensures Gallery is NOT always the same images per person,
        eliminating systematic bias that could inflate or deflate metrics.
        """
        from collections import defaultdict
        person_to_samples = defaultdict(list)
        for sample in samples:
            person_to_samples[sample['face_id']].append(sample)

        # 固定随机种子，保证 Gallery/Query 划分可复现
        # 使用 seed+200 与 train/val split 的 seed 区分开
        random.seed(self.seed + 200)

        gallery_paths = []
        gallery_labels = []
        gallery_person_ids = []

        query_paths = []
        query_labels = []
        query_person_ids = []

        for person_id, person_samples in person_to_samples.items():
            shuffled = list(person_samples)
            random.shuffle(shuffled)
            gallery = shuffled[:self.gallery_per_person]
            query = shuffled[self.gallery_per_person:]

            for s in gallery:
                gallery_paths.append((s['face_path'], s['fingerprint_path']))
                gallery_labels.append(self.class_to_idx[s['label']])
                gallery_person_ids.append(person_id)

            for s in query:
                query_paths.append((s['face_path'], s['fingerprint_path']))
                query_labels.append(self.class_to_idx[s['label']])
                query_person_ids.append(person_id)

        setattr(self, f'{split_name}_gallery_paths', gallery_paths)
        setattr(self, f'{split_name}_gallery_labels', gallery_labels)
        setattr(self, f'{split_name}_gallery_person_ids', gallery_person_ids)
        setattr(self, f'{split_name}_query_paths', query_paths)
        setattr(self, f'{split_name}_query_labels', query_labels)
        setattr(self, f'{split_name}_query_person_ids', query_person_ids)

        _logger.info(
            f"[{split_name.capitalize()} Gallery/Query] Gallery: {len(gallery_paths)} pairs "
            f"({len(person_to_samples)} persons × {self.gallery_per_person}), "
            f"Query: {len(query_paths)} pairs"
        )

    def get_query_loader(self, batch_size=32, num_workers=0):
        """返回一个只包含 Query 样本的 DataLoader。

        Gallery 和 Query 来自不同的人（Gallery 样本不参与 Query 检索），
        因此验证时必须分别提取两者的特征，再计算相似度矩阵。
        本方法支持 Gallery/Query 分离的检索场景（如消融验证）。
        """
        query_paths = getattr(self, f'{self.mode}_query_paths', None)
        query_labels = getattr(self, f'{self.mode}_query_labels', None)
        if not query_paths:
            return None

        query_samples = []
        for (face_path, fp_path), label in zip(query_paths, query_labels):
            query_samples.append({
                'face_path': face_path,
                'fingerprint_path': fp_path,
                'label': label,
            })

        class _QueryDataset(Dataset):
            def __init__(self, samples, face_transform, fp_transform):
                self.samples = samples
                self.face_transform = face_transform
                self.fp_transform = fp_transform

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                s = self.samples[idx]
                face_img = Image.open(s['face_path']).convert('RGB')
                fp_img = Image.open(s['fingerprint_path']).convert('RGB')
                if self.face_transform:
                    face_img = self.face_transform(face_img)
                if self.fp_transform:
                    fp_img = self.fp_transform(fp_img)
                return {
                    'face_image': face_img,
                    'fingerprint_image': fp_img,
                    'label': s['label'],
                }

        face_t = self.transform.get('face_transform') if isinstance(self.transform, dict) else self.transform
        fp_t = self.transform.get('fp_transform') if isinstance(self.transform, dict) else self.transform

        return DataLoader(
            _QueryDataset(query_samples, face_t, fp_t),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def _get_default_transform(self):
        """获取默认数据变换"""
        aug_params = self.augmentation_params or {}

        if self.mode == 'train' and self.augment:
            # Face transform
            face_transform_list = []
            if aug_params.get("random_resized_crop", False):
                face_transform_list.append(transforms.RandomResizedCrop(self.face_image_size))
            else:
                face_transform_list.append(transforms.Resize((self.face_image_size, self.face_image_size)))

            if aug_params.get("random_horizontal_flip", True):
                face_transform_list.append(transforms.RandomHorizontalFlip())
            if aug_params.get("random_rotation", 10):
                face_transform_list.append(transforms.RandomRotation(aug_params.get("random_rotation", 10)))
            if aug_params.get("color_jitter", False):
                face_transform_list.append(transforms.ColorJitter(
                    brightness=aug_params.get("color_jitter_brightness", 0.2),
                    contrast=aug_params.get("color_jitter_contrast", 0.2),
                    saturation=aug_params.get("color_jitter_saturation", 0.1),
                    hue=aug_params.get("color_jitter_hue", 0.05),
                ))

            face_transform_list.append(transforms.ToTensor())
            re_prob = float(aug_params.get("random_erasing_prob", 0.0) or 0.0)
            if re_prob > 0:
                face_transform_list.append(transforms.RandomErasing(p=re_prob))
            face_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

            face_transform = transforms.Compose(face_transform_list)

            # Fingerprint transform
            fp_transform_list = []
            fp_transform_list.append(transforms.Resize((self.fingerprint_image_size, self.fingerprint_image_size)))

            if aug_params.get("random_rotation", 5):
                rot_degrees = int(aug_params.get("random_rotation", 5))
                fp_transform_list.append(transforms.RandomRotation(rot_degrees))

            if aug_params.get("gaussian_blur", True):
                fp_transform_list.append(transforms.GaussianBlur(kernel_size=3))

            fp_transform_list.append(transforms.ToTensor())
            if re_prob > 0:
                fp_transform_list.append(transforms.RandomErasing(p=re_prob))
            # 统一使用 ImageNet 归一化（与 FingerprintDataset 和推理 pipeline 一致）
            fp_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

            fp_transform = transforms.Compose(fp_transform_list)
        else:
            face_transform = transforms.Compose([
                transforms.Resize((self.face_image_size, self.face_image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            fp_transform = transforms.Compose([
                transforms.Resize((self.fingerprint_image_size, self.fingerprint_image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        return {'face': face_transform, 'fingerprint': fp_transform}

    def _apply_clahe(self, img: Image.Image) -> Image.Image:
        """对指纹图像应用 CLAHE 增强（每个 worker 独立创建 CLAHE 对象）。"""
        if not self.use_clahe:
            return img
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        if not hasattr(self, '_clahe_cached'):
            self._clahe_cached = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
        enhanced = self._clahe_cached.apply(gray)
        return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            face_image = Image.open(sample['face_path']).convert('RGB')
        except Exception as e:
            print(f"加载人脸图像失败 {sample['face_path']}: {e}")
            face_image = Image.new('RGB', (self.face_image_size, self.face_image_size), (128, 128, 128))

        try:
            fp_image = Image.open(sample['fingerprint_path']).convert('RGB')
            fp_image = self._apply_clahe(fp_image)
        except Exception as e:
            print(f"加载指纹图像失败 {sample['fingerprint_path']}: {e}")
            fp_image = Image.new('RGB', (self.fingerprint_image_size, self.fingerprint_image_size), (128, 128, 128))

        if isinstance(self.transform, dict):
            face_image = self.transform['face'](face_image)
            fp_image = self.transform['fingerprint'](fp_image)
        else:
            face_image = self.transform(face_image)
            fp_image = self.transform(fp_image)

        return {
            'face_image': face_image,
            'fingerprint_image': fp_image,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'face_path': sample['face_path'],
            'fingerprint_path': sample['fingerprint_path'],
            'face_id': sample['face_id'],
            'fingerprint_id': sample['fingerprint_id']
        }
