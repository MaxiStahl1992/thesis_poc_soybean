import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset, WeightedRandomSampler
from torchvision import transforms
from .soybean import SoybeanDataset
import numpy as np


def compute_class_weights(dataset, num_classes=3):
    """
    Compute inverse frequency class weights for balanced sampling/loss.
    
    Args:
        dataset: PyTorch dataset with labels
        num_classes: Number of classes
        
    Returns:
        class_weights: Tensor of shape (num_classes,) with inverse frequency weights
        class_counts: Tensor of shape (num_classes,) with sample counts per class
    """
    # Count samples per class
    class_counts = torch.zeros(num_classes)
    
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_counts[label] += 1
    
    # Compute inverse frequency weights
    total_samples = len(dataset)
    class_weights = total_samples / (num_classes * class_counts)
    
    # Normalize weights to sum to num_classes (for numerical stability)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print(f"📊 Class distribution:")
    for i in range(num_classes):
        print(f"   Class {i}: {int(class_counts[i])} samples (weight: {class_weights[i]:.3f})")
    
    return class_weights, class_counts


def get_weighted_sampler(dataset, class_weights):
    """
    Create WeightedRandomSampler for balanced class sampling during TTA.
    
    Args:
        dataset: PyTorch dataset
        class_weights: Tensor of class weights from compute_class_weights
        
    Returns:
        sampler: WeightedRandomSampler that oversamples minority classes
    """
    # Assign weight to each sample based on its class
    sample_weights = torch.zeros(len(dataset))
    
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        sample_weights[idx] = class_weights[label]
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True  # Allow oversampling
    )
    
    return sampler


def get_dataloaders(dataset_name, data_root, batch_size=32, train_val_test_split=(0.7, 0.15, 0.15), 
                    seed=21, train_transform=None, test_transform=None):
    """
    Creates DataLoaders for train, val, and test splits.
    
    Args:
        dataset_name: Name of the dataset ('ASDID', 'MH', etc.)
        data_root: Root directory of the dataset
        batch_size: Batch size for dataloaders
        train_val_test_split: Tuple of (train, val, test) split ratios
        seed: Random seed for reproducibility
        train_transform: Custom transforms for training data (if None, uses default)
        test_transform: Custom transforms for val/test data (if None, uses default)
    
    Returns:
        (train_loader, val_loader, test_loader, dataset, train_indices, val_indices, test_indices)
    """
    # Standard ConNeXt/Swin/ResNet transforms (default)
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Create dataset with test transform initially (for splitting)
    dataset = SoybeanDataset(data_root, dataset_name, transform=test_transform)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_val_test_split[0] * total_size)
    val_size = int(train_val_test_split[1] * total_size)
    test_size = total_size - train_size - val_size
    
    # Deterministic split
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    
    # Apply train_transform to training subset if different from test_transform
    if train_transform is not test_transform:
        # Create a new dataset with train transforms
        train_dataset = SoybeanDataset(data_root, dataset_name, transform=train_transform)
        # Use Subset with train indices on the train_dataset
        from torch.utils.data import Subset
        train_ds = Subset(train_dataset, train_ds.indices)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, dataset, train_ds.indices, val_ds.indices, test_ds.indices


def get_tta_dataloaders(dataset_name, data_root, batch_size=32, 
                        use_early_stopping=False, val_split=0.2, seed=21,
                        use_weighted_sampling=False, num_classes=3):
    """
    Creates DataLoaders for TTA: combines ALL unlabeled target domain data.
    
    For TTA, we assume ALL target domain data is unlabeled, so we combine
    what would normally be train/val/test splits into a single dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'MH')
        data_root: Root directory of the dataset
        batch_size: Batch size for dataloaders
        use_early_stopping: If True, split combined data into adapt/val for early stopping
        val_split: Fraction of data to reserve for validation (only if use_early_stopping=True)
        seed: Random seed for reproducibility
        use_weighted_sampling: If True, use weighted sampling to balance classes
        num_classes: Number of classes for weight computation
        
    Returns:
        If use_early_stopping=False:
            - adapt_loader: DataLoader with ALL unlabeled target data
            - None: (placeholder for consistency)
            - class_weights: Tensor of class weights (if use_weighted_sampling else None)
            
        If use_early_stopping=True:
            - adapt_loader: DataLoader with adaptation subset
            - val_loader: DataLoader with validation subset
            - class_weights: Tensor of class weights (if use_weighted_sampling else None)
    """
    # Standard ConNeXt/Swin/ResNet transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load the entire dataset (treating all as unlabeled)
    full_dataset = SoybeanDataset(data_root, dataset_name, transform=transform)
    
    # Compute class weights if needed
    class_weights = None
    if use_weighted_sampling:
        class_weights, _ = compute_class_weights(full_dataset, num_classes)
    
    if not use_early_stopping:
        # Use ALL data for adaptation (standard single-pass TTA)
        if use_weighted_sampling:
            sampler = get_weighted_sampler(full_dataset, class_weights)
            adapt_loader = DataLoader(
                full_dataset, 
                batch_size=batch_size, 
                sampler=sampler  # Use weighted sampler
            )
        else:
            adapt_loader = DataLoader(
                full_dataset, 
                batch_size=batch_size, 
                shuffle=True  # Shuffle for better adaptation
            )
        return adapt_loader, None, class_weights
    
    else:
        # Split into adaptation and validation sets for early stopping
        total_size = len(full_dataset)
        val_size = int(val_split * total_size)
        adapt_size = total_size - val_size
        
        # Deterministic split
        generator = torch.Generator().manual_seed(seed)
        adapt_ds, val_ds = random_split(
            full_dataset, 
            [adapt_size, val_size], 
            generator=generator
        )
        
        # Create adapt loader (with or without weighted sampling)
        if use_weighted_sampling:
            # Create weighted sampler for adapt_ds
            adapt_sample_weights = torch.zeros(len(adapt_ds))
            for idx in range(len(adapt_ds)):
                _, label = adapt_ds[idx]
                adapt_sample_weights[idx] = class_weights[label]
            
            adapt_sampler = WeightedRandomSampler(
                weights=adapt_sample_weights,
                num_samples=len(adapt_ds),
                replacement=True
            )
            adapt_loader = DataLoader(adapt_ds, batch_size=batch_size, sampler=adapt_sampler)
        else:
            adapt_loader = DataLoader(adapt_ds, batch_size=batch_size, shuffle=True)
        
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        print(f"📊 TTA Data Split: {adapt_size} samples for adaptation, {val_size} samples for validation")
        
        return adapt_loader, val_loader, class_weights
