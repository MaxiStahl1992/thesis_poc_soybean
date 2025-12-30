"""
Segmentation Module
==================

Utilities for background removal using SAM (Segment Anything Model).
"""

from .sam_utils import load_sam_model, load_sam_semantic_predictor, segment_image_with_text, segment_image_with_bbox, segment_image, get_largest_mask, get_all_masks, visualize_masks, get_mask_statistics
from .background_removal import remove_background, remove_background_with_alpha, process_image_with_sam, apply_morphological_operations, smooth_mask_edges, visualize_background_removal
from .batch_processing import process_dataset, create_segmented_dataset, compare_original_vs_segmented, verify_dataset_structure

__all__ = [
    'load_sam_model',
    'load_sam_semantic_predictor'
    'segment_image_with_text',
    'segment_image_with_bbox',
    'segment_image',
    'get_largest_mask',
    'get_all_masks',
    'visualize_masks',
    'get_mask_statistics',
    'remove_background',
    'remove_background_with_alpha',
    'process_image_with_sam',
    'apply_morphological_operations',
    'smooth_mask_edges',
    'visualize_background_removal',
    'process_dataset',
    'create_segmented_dataset',
    'compare_original_vs_segmented',
    'verify_dataset_structure'
]
