# Datasets module initialization
# This module provides unified access to all dataset classes

from .face_dataset import FaceDataset
from .fusion_dataset import FusionDataset

__all__ = [
    'FaceDataset',
    'FusionDataset',
]