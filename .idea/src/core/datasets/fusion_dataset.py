import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
import pandas as pd


class FusionDataset(Dataset):
    """多模态人脸+指纹融合数据集

    支持真正的人脸-指纹配对数据加载
    """

    def __init__(self, face_data_dir, fingerprint_data_dir,
                 mapping_file=None, mode='train',
                 face_image_size=224, fingerprint_image_size=224,
                 transform=None, augment=True):
        """
        Args:
            face_data_dir: 人脸数据目录路径
            fingerprint_data_dir: 指纹数据目录路径
            mapping_file: 人脸-指纹映射文件路径 (CSV或JSON)
            mode: 'train' 或 'val'
            face_image_size: 人脸图像尺寸
            fingerprint_image_size: 指纹图像尺寸
            transform: 自定义变换
            augment: 是否使用数据增强
        """
        self.face_data_dir = face_data_dir
        self.fingerprint_data_dir = fingerprint_data_dir
        # 从 face_data_dir 提取 data_dir (data/face/face -> data)
        self.data_dir = os.path.dirname(os.path.dirname(face_data_dir))
        self.mapping_file = mapping_file
        self.mode = mode
        self.face_image_size = face_image_size
        self.fingerprint_image_size = fingerprint_image_size
        self.augment = augment
        # optional augmentation parameters for both modalities
        self.augmentation_params = None

        # 加载人脸-指纹映射
        if mapping_file:
            self.face_fp_mapping = self._load_mapping(mapping_file)
        else:
            # 如果没有映射文件，使用目录名匹配
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
            # 假设CSV格式: face_id,fingerprint_id,class_label
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
                # 转换JSON格式为标准格式
                mapping = {}
                for face_id, info in data.items():
                    person_id = info['person_id']
                    fp_id = info['fingerprint_id']
                    # 使用person_id作为标签（需要转换为数字）
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

        # 假设人脸和指纹目录使用相同的ID命名
        if os.path.exists(self.face_data_dir):
            face_dirs = [d for d in os.listdir(self.face_data_dir)
                        if os.path.isdir(os.path.join(self.face_data_dir, d))]

            for face_id in face_dirs:
                # 假设指纹ID与人脸ID相同
                mapping[face_id] = {
                    'fingerprint_id': face_id,
                    'label': len(mapping)  # 使用索引作为标签
                }

        return mapping

    def _collect_samples(self):
        """收集所有有效的样本"""
        for face_id, info in self.face_fp_mapping.items():
            fp_id = info['fingerprint_id']
            label = info['label']

            # 从映射文件中获取图像路径
            face_image_paths_raw = info.get('face_images', [])
            fingerprint_images = info.get('fingerprint_images', {})

            # 转换人脸图像路径为绝对路径
            face_image_paths = []
            for img_path in face_image_paths_raw:
                # JSON中的路径是相对于data目录的
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
                # 从映射文件中获取（需要转换为绝对路径）
                for hand, images in fingerprint_images.items():
                    if isinstance(images, list):
                        for img_path in images:
                            # JSON中的路径是相对于data目录的
                            abs_path = os.path.join(self.data_dir, img_path.replace('\\', os.sep))
                            fp_image_paths.append(abs_path)
            else:
                # 从目录中扫描
                fp_base_dir = os.path.join(self.fingerprint_data_dir, fp_id)
                if os.path.exists(fp_base_dir):
                    for hand in ['left', 'right']:
                        hand_dir = os.path.join(fp_base_dir, hand)
                        if os.path.exists(hand_dir):
                            for img_name in os.listdir(hand_dir):
                                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                    fp_image_paths.append(os.path.join(hand_dir, img_name))

            # 为每个人脸图像配对指纹图像
            if face_image_paths and fp_image_paths:
                num_pairs = min(len(face_image_paths), len(fp_image_paths))
                for i in range(num_pairs):
                    fp_idx = i % len(fp_image_paths)  # 循环使用指纹图像
                    self.samples.append({
                        'face_path': face_image_paths[i],
                        'fingerprint_path': fp_image_paths[fp_idx],
                        'face_id': face_id,
                        'fingerprint_id': fp_id,
                        'label': label
                    })

        # 创建类别映射
        unique_labels = sorted(set(sample['label'] for sample in self.samples))
        self.class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        # 更新样本标签为连续索引
        for sample in self.samples:
            sample['label'] = self.class_to_idx[sample['label']]

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
                face_transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))

            face_transform_list.append(transforms.ToTensor())
            re_prob = float(aug_params.get("random_erasing_prob", 0.0) or 0.0)
            if re_prob > 0:
                face_transform_list.append(transforms.RandomErasing(p=re_prob))
            face_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

            face_transform = transforms.Compose(face_transform_list)

            # Fingerprint transform
            fp_transform_list = []
            if aug_params.get("random_resized_crop", False):
                fp_transform_list.append(transforms.RandomResizedCrop(self.fingerprint_image_size))
            else:
                fp_transform_list.append(transforms.Resize((self.fingerprint_image_size, self.fingerprint_image_size)))

            if aug_params.get("random_horizontal_flip", True):
                fp_transform_list.append(transforms.RandomHorizontalFlip())
            if aug_params.get("random_rotation", 5):
                fp_transform_list.append(transforms.RandomRotation(aug_params.get("random_rotation", 5)))
            if aug_params.get("color_jitter", False):
                fp_transform_list.append(transforms.ColorJitter(brightness=0.05, contrast=0.05))

            fp_transform_list.append(transforms.ToTensor())
            if re_prob > 0:
                fp_transform_list.append(transforms.RandomErasing(p=re_prob))
            fp_transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

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
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        return {'face': face_transform, 'fingerprint': fp_transform}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载人脸图像
        try:
            face_image = Image.open(sample['face_path']).convert('RGB')
        except Exception as e:
            print(f"加载人脸图像失败 {sample['face_path']}: {e}")
            face_image = Image.new('RGB', (self.face_image_size, self.face_image_size), (128, 128, 128))

        # 加载指纹图像
        try:
            fp_image = Image.open(sample['fingerprint_path']).convert('RGB')
        except Exception as e:
            print(f"加载指纹图像失败 {sample['fingerprint_path']}: {e}")
            fp_image = Image.new('RGB', (self.fingerprint_image_size, self.fingerprint_image_size), (128, 128, 128))

        # 应用变换
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