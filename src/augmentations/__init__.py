"""
Augmentation pipelines for domain generalization experiments.
"""

from .transforms import (
    get_baseline_transforms,
    get_dg_transforms,
    get_yolo_baseline_augmentations,
    get_yolo_dg_augmentations
)

__all__ = [
    'get_baseline_transforms',
    'get_dg_transforms',
    'get_yolo_baseline_augmentations',
    'get_yolo_dg_augmentations'
]
