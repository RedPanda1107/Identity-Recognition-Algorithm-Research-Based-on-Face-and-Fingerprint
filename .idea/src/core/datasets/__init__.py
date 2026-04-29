# Datasets module initialization
# This module provides unified access to all dataset classes

from .base_dataset import BaseDataset
from .face_dataset import FaceDataset
from .fingerprint_dataset import FingerprintDataset
from .fusion_dataset import FusionDataset

__all__ = [
    'BaseDataset',
    'FaceDataset',
    'FingerprintDataset',
    'FusionDataset',
]