"""Base dataset class providing unified person-wise split logic.

All multimodal datasets (Face, Fingerprint, Fusion) should inherit from this class
to ensure consistent:
- Person-wise train/val splitting
- Gallery/Query separation for open-set evaluation
- Random seed control for reproducibility
"""

import os
import random
import logging
from abc import ABC, abstractmethod

_logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    """Abstract base class for person-level identification datasets.

    Provides unified logic for:
    - Person-wise train/val split (no person overlap)
    - Gallery/Query separation for validation
    - Reproducible splits via random seed

    Subclasses must implement:
    - _collect_person_samples(): Collect all images grouped by person_id
    - _get_transform(): Define image transformations
    - __getitem__(): Return image tensor and label
    """

    DEFAULT_SPLIT_RATIO = 0.8
    DEFAULT_GALLERY_PER_PERSON = 3
    DEFAULT_SEED = 42

    def __init__(
        self,
        data_dir: str,
        mode: str = 'train',
        image_size: int = 224,
        augment: bool = True,
        split_ratio: float = None,
        gallery_per_person: int = None,
        seed: int = None,
        class_to_idx: dict = None,
        **kwargs
    ):
        """
        Args:
            data_dir: Path to data directory
            mode: 'train' or 'val'
            image_size: Target image size
            augment: Whether to apply data augmentation
            split_ratio: Person-wise train/val split ratio (default: 0.8)
            gallery_per_person: Number of gallery images per person (default: 3)
            seed: Random seed for reproducible splits (default: 42)
            class_to_idx: Optional label mapping for train/val consistency
        """
        self.data_dir = data_dir
        self.mode = mode
        self.image_size = image_size
        self.augment = augment

        # Unified parameters with defaults
        self.split_ratio = split_ratio if split_ratio is not None else self.DEFAULT_SPLIT_RATIO
        self.gallery_per_person = gallery_per_person if gallery_per_person is not None else self.DEFAULT_GALLERY_PER_PERSON
        self.seed = seed if seed is not None else self.DEFAULT_SEED

        self._provided_class_to_idx = class_to_idx
        self.augmentation_params = None

        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.person_id_to_idx = {}

        self._load_data()
        self.transform = self._get_transform()

    def _load_data(self):
        """Main loading logic. Collect samples, split, create gallery/query."""
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        # Subclass collects all images grouped by person_id
        person_samples = self._collect_person_samples()

        # Build label mapping
        self._build_label_mapping(person_samples)

        # Person-wise train/val split
        train_samples, val_samples = self._person_wise_split(person_samples)

        # Validate split integrity
        self._validate_split(train_samples, val_samples)

        # Create Gallery/Query for validation
        self._create_gallery_query_split(val_samples)

        # Select samples for current mode
        if self.mode == 'train':
            filtered_samples = train_samples
        else:
            filtered_samples = val_samples

        self.image_paths = [s['img_path'] for s in filtered_samples]
        self.labels = [s['label_idx'] for s in filtered_samples]

        _logger.info(f"{self.__class__.__name__} mode={self.mode}, "
                     f"samples={len(self.image_paths)}, classes={len(self.class_to_idx)}")

    @abstractmethod
    def _collect_person_samples(self) -> dict:
        """Collect all images grouped by person_id.

        Returns:
            dict: {person_id: [img_path1, img_path2, ...]}
        """
        pass

    def _build_label_mapping(self, person_samples: dict):
        """Build person_id -> class_index mapping."""
        if self._provided_class_to_idx is not None:
            self.person_id_to_idx = self._provided_class_to_idx
            self.class_to_idx = self._provided_class_to_idx
            # Filter to only persons in mapping
            person_samples.update({
                pid: paths for pid, paths in person_samples.items()
                if pid in self.person_id_to_idx
            })
        else:
            sorted_person_ids = sorted(person_samples.keys())
            self.person_id_to_idx = {pid: idx for idx, pid in enumerate(sorted_person_ids)}
            self.class_to_idx = self.person_id_to_idx

    def _person_wise_split(self, person_samples: dict) -> tuple:
        """Split persons into train/val sets (no overlap).

        Returns:
            tuple: (train_samples, val_samples)
        """
        all_person_ids = list(person_samples.keys())

        # Set seed for reproducibility
        random.seed(self.seed)
        random.shuffle(all_person_ids)

        split_idx = int(len(all_person_ids) * self.split_ratio)
        train_person_ids = set(all_person_ids[:split_idx])
        val_person_ids = set(all_person_ids[split_idx:])

        train_samples = []
        val_samples = []

        for person_id, img_paths in person_samples.items():
            person_idx = self.person_id_to_idx[person_id]
            for img_path in img_paths:
                sample = {
                    'person_id': person_id,
                    'img_path': img_path,
                    'label_idx': person_idx
                }
                if person_id in train_person_ids:
                    train_samples.append(sample)
                else:
                    val_samples.append(sample)

        return train_samples, val_samples

    def _validate_split(self, train_samples: list, val_samples: list):
        """Validate that train/val sets are truly disjoint."""
        train_person_set = set(s['person_id'] for s in train_samples)
        val_person_set = set(s['person_id'] for s in val_samples)

        person_overlap = train_person_set & val_person_set
        if person_overlap:
            _logger.error(f"[ERROR] Person overlap detected: {person_overlap}")
            raise RuntimeError("Data leakage: persons appear in both train and val sets")
        else:
            _logger.info(f"[PASS] Person-wise split: train={len(train_person_set)} persons, "
                         f"val={len(val_person_set)} persons (no overlap)")

        train_paths_set = set(s['img_path'] for s in train_samples)
        val_paths_set = set(s['img_path'] for s in val_samples)
        path_overlap = train_paths_set & val_paths_set
        if path_overlap:
            _logger.error(f"[ERROR] Path overlap: {len(path_overlap)} images in both sets")
            raise RuntimeError("Data leakage: images appear in both train and val sets")
        else:
            _logger.info("[PASS] No image path overlap between train and val sets")

        _logger.info(f"Train: {len(train_samples)} images / {len(train_person_set)} persons")
        _logger.info(f"Val:   {len(val_samples)} images / {len(val_person_set)} persons")

    def _create_gallery_query_split(self, val_samples: list):
        """Create Gallery/Query split for validation (1:N retrieval).

        Gallery: First N images per validation person
        Query: Remaining validation images (excluding gallery)
        """
        # Group val samples by person
        person_to_paths = {}
        for sample in val_samples:
            pid = sample['person_id']
            if pid not in person_to_paths:
                person_to_paths[pid] = []
            person_to_paths[pid].append(sample)

        self.val_gallery_paths = []
        self.val_gallery_labels = []
        self.val_gallery_person_ids = []

        self.val_query_paths = []
        self.val_query_labels = []
        self.val_query_person_ids = []

        for person_id, samples in person_to_paths.items():
            person_idx = self.person_id_to_idx[person_id]

            # Sort by path for consistency
            sorted_samples = sorted(samples, key=lambda x: x['img_path'])

            gallery = sorted_samples[:self.gallery_per_person]
            query = sorted_samples[self.gallery_per_person:]

            for s in gallery:
                self.val_gallery_paths.append(s['img_path'])
                self.val_gallery_labels.append(s['label_idx'])
                self.val_gallery_person_ids.append(person_id)

            for s in query:
                self.val_query_paths.append(s['img_path'])
                self.val_query_labels.append(s['label_idx'])
                self.val_query_person_ids.append(person_id)

        _logger.info(f"Val Gallery: {len(self.val_gallery_paths)} images "
                     f"({len(person_to_paths)} persons × {self.gallery_per_person})")
        _logger.info(f"Val Query:   {len(self.val_query_paths)} images "
                     f"(remaining validation images)")

    @abstractmethod
    def _get_transform(self):
        """Return image transformation pipeline."""
        pass

    def __len__(self):
        return len(self.image_paths)

    @abstractmethod
    def __getitem__(self, idx):
        """Return image tensor, label, and path."""
        pass

    def get_gallery_query_info(self) -> dict:
        """Return gallery/query sizes for logging."""
        return {
            'gallery_size': len(self.val_gallery_paths),
            'query_size': len(self.val_query_paths)
        }
