"""
Few-Shot Learning Module
========================

Utilities for few-shot linear probing and domain adaptation.
"""

from .sampling import create_k_shot_subset, stratified_sample_indices, analyze_k_shot_distribution, create_multiple_k_shot_subsets
from .linear_probe import freeze_backbone, linear_probe_finetune, evaluate_linear_probe, get_trainable_param_count
from .visualization import plot_return_on_labeling, plot_confusion_comparison, plot_rust_frogeye_confusion, plot_training_curves

__all__ = [
    'create_k_shot_subset',
    'stratified_sample_indices',
    'analyze_k_shot_distribution',
    'create_multiple_k_shot_subsets',
    'freeze_backbone',
    'linear_probe_finetune',
    'evaluate_linear_probe',
    'get_trainable_param_count',
    'plot_return_on_labeling',
    'plot_confusion_comparison',
    'plot_rust_frogeye_confusion',
    'plot_training_curves',
]
