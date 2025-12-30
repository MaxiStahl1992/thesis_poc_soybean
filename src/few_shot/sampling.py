"""
K-Shot Sampling Utilities
=========================

Functions for creating stratified k-shot subsets from datasets.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from typing import List, Dict, Tuple
from collections import defaultdict


def stratified_sample_indices(dataset: Dataset, k_shot: int, num_classes: int, seed: int = 42) -> List[int]:
    """
    Sample k examples per class from a dataset in a stratified manner.
    
    Args:
        dataset: PyTorch dataset with labels accessible via dataset[i][1]
        k_shot: Number of examples to sample per class
        num_classes: Total number of classes in the dataset
        seed: Random seed for reproducibility
        
    Returns:
        List of indices for the k-shot subset
        
    Example:
        >>> indices = stratified_sample_indices(train_dataset, k_shot=5, num_classes=3)
        >>> k_shot_subset = Subset(train_dataset, indices)
    """
    np.random.seed(seed)
    
    # Group indices by class
    class_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if isinstance(label, torch.Tensor):
            label = label.item()
        class_to_indices[label].append(idx)
    
    # Sample k indices per class
    selected_indices = []
    for class_id in range(num_classes):
        if class_id not in class_to_indices:
            raise ValueError(f"Class {class_id} not found in dataset")
        
        available_indices = class_to_indices[class_id]
        if len(available_indices) < k_shot:
            raise ValueError(
                f"Class {class_id} has only {len(available_indices)} examples, "
                f"but k_shot={k_shot} requested"
            )
        
        # Random sample without replacement
        sampled = np.random.choice(available_indices, size=k_shot, replace=False)
        selected_indices.extend(sampled.tolist())
    
    # Shuffle to mix classes
    np.random.shuffle(selected_indices)
    
    return selected_indices


def create_k_shot_subset(dataset: Dataset, k_shot: int, num_classes: int, seed: int = 42) -> Subset:
    """
    Create a k-shot subset from a dataset.
    
    Args:
        dataset: PyTorch dataset
        k_shot: Number of examples per class
        num_classes: Total number of classes
        seed: Random seed
        
    Returns:
        Subset containing k examples per class
        
    Example:
        >>> k_shot_data = create_k_shot_subset(full_dataset, k_shot=5, num_classes=3)
        >>> print(f"Total samples: {len(k_shot_data)}")  # 15 (5 per class × 3 classes)
    """
    indices = stratified_sample_indices(dataset, k_shot, num_classes, seed)
    return Subset(dataset, indices)


def analyze_k_shot_distribution(subset: Subset, num_classes: int) -> Dict[int, int]:
    """
    Analyze the class distribution in a k-shot subset.
    
    Args:
        subset: K-shot subset to analyze
        num_classes: Total number of classes
        
    Returns:
        Dictionary mapping class_id -> count
        
    Example:
        >>> distribution = analyze_k_shot_distribution(k_shot_subset, num_classes=3)
        >>> print(distribution)  # {0: 5, 1: 5, 2: 5}
    """
    class_counts = defaultdict(int)
    
    for idx in range(len(subset)):
        _, label = subset[idx]
        if isinstance(label, torch.Tensor):
            label = label.item()
        class_counts[label] += 1
    
    # Ensure all classes are represented
    for class_id in range(num_classes):
        if class_id not in class_counts:
            class_counts[class_id] = 0
    
    return dict(sorted(class_counts.items()))


def create_multiple_k_shot_subsets(
    dataset: Dataset,
    k_shots: List[int],
    num_classes: int,
    seed: int = 42
) -> Dict[int, Subset]:
    """
    Create multiple k-shot subsets with different k values.
    
    Args:
        dataset: Source dataset
        k_shots: List of k values (e.g., [5, 10, 15])
        num_classes: Number of classes
        seed: Random seed
        
    Returns:
        Dictionary mapping k -> Subset
        
    Example:
        >>> subsets = create_multiple_k_shot_subsets(dataset, k_shots=[5, 10, 15], num_classes=3)
        >>> for k, subset in subsets.items():
        ...     print(f"k={k}: {len(subset)} samples")
    """
    subsets = {}
    for k in k_shots:
        subsets[k] = create_k_shot_subset(dataset, k_shot=k, num_classes=num_classes, seed=seed)
    return subsets


def get_remaining_indices(dataset: Dataset, k_shot_indices: List[int]) -> List[int]:
    """
    Get all indices NOT in the k-shot subset (for validation/test).
    
    Args:
        dataset: Full dataset
        k_shot_indices: Indices used for k-shot training
        
    Returns:
        List of remaining indices
        
    Example:
        >>> train_indices = stratified_sample_indices(dataset, k_shot=5, num_classes=3)
        >>> val_indices = get_remaining_indices(dataset, train_indices)
    """
    all_indices = set(range(len(dataset)))
    k_shot_set = set(k_shot_indices)
    remaining = list(all_indices - k_shot_set)
    return sorted(remaining)
