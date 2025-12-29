"""
Augmentation transforms for baseline and domain generalization (DG) training.

Baseline augmentations:
    - Standard ImageNet-style augmentations
    - Mild augmentations for typical supervised learning
    
DG augmentations (Style Randomization):
    - Heavy augmentations designed to expand source distribution
    - Inspired by domain generalization research
    - Goal: make target domain fall within augmented source distribution
"""

import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import numpy as np
from PIL import ImageFilter, ImageEnhance
import random


# ============================================================================
# PyTorch Transforms (for DINOv2, ResNet, ViT)
# ============================================================================

def get_baseline_transforms(img_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Baseline augmentations for standard supervised learning.
    
    Similar to typical ImageNet training augmentations:
    - Resize + RandomCrop
    - Horizontal flip
    - Mild color jitter
    - Normalization
    
    Args:
        img_size: Target image size
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        torchvision.transforms.Compose
    """
    return T.Compose([
        T.Resize(int(img_size * 1.14)),  # Resize to slightly larger
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])


def get_dg_transforms(img_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Domain Generalization (DG) augmentations with style randomization.
    
    Heavy augmentations to expand source distribution:
    - Strong color jitter (brightness, contrast, saturation, hue)
    - Gaussian blur with varying kernel sizes
    - Random sharpening
    - Aspect ratio jittering (RandomResizedCrop)
    - Random rotation
    - Random erasing (cutout)
    - JPEG compression artifacts
    - Random grayscale conversion
    
    These augmentations aim to simulate various domain shifts that might
    occur in the target domain (different lighting, camera quality, etc.).
    
    Args:
        img_size: Target image size
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        torchvision.transforms.Compose
    """
    return T.Compose([
        # Aggressive spatial augmentations
        T.RandomResizedCrop(
            img_size, 
            scale=(0.7, 1.0),  # More aggressive cropping
            ratio=(0.75, 1.33),  # Aspect ratio jittering
            interpolation=InterpolationMode.BILINEAR
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),  # Small rotations
        
        # Strong color augmentations
        T.ColorJitter(
            brightness=0.4,  # 2x stronger than baseline
            contrast=0.4,
            saturation=0.4,
            hue=0.2
        ),
        
        # Style augmentations
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        T.RandomGrayscale(p=0.1),  # Occasionally remove color info
        
        # Convert to tensor
        T.ToTensor(),
        
        # Additional tensor-level augmentations
        T.RandomApply([
            T.Lambda(lambda x: add_jpeg_artifacts(x))
        ], p=0.2),
        
        T.RandomApply([
            T.Lambda(lambda x: sharpen_image(x))
        ], p=0.2),
        
        # Random erasing (cutout)
        T.RandomErasing(
            p=0.3,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3),
            value='random'
        ),
        
        # Normalization
        T.Normalize(mean=mean, std=std)
    ])


# ============================================================================
# Helper functions for custom augmentations
# ============================================================================

def add_jpeg_artifacts(tensor):
    """
    Simulate JPEG compression artifacts.
    
    Args:
        tensor: Image tensor (C, H, W) in [0, 1]
    
    Returns:
        Tensor with JPEG artifacts
    """
    from io import BytesIO
    from PIL import Image
    
    # Convert to PIL
    img = T.ToPILImage()(tensor)
    
    # Save with low quality and reload
    buffer = BytesIO()
    quality = random.randint(40, 80)  # Varying quality
    img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    img = Image.open(buffer)
    
    # Convert back to tensor
    return T.ToTensor()(img)


def sharpen_image(tensor):
    """
    Apply random sharpening to image.
    
    Args:
        tensor: Image tensor (C, H, W) in [0, 1]
    
    Returns:
        Sharpened tensor
    """
    from PIL import Image, ImageEnhance
    
    # Convert to PIL
    img = T.ToPILImage()(tensor)
    
    # Apply sharpening with random strength
    enhancer = ImageEnhance.Sharpness(img)
    factor = random.uniform(1.5, 2.5)  # Sharpening factor
    img = enhancer.enhance(factor)
    
    # Convert back to tensor
    return T.ToTensor()(img)


# ============================================================================
# YOLO Augmentations (dict format for ultralytics)
# ============================================================================

def get_yolo_baseline_augmentations():
    """
    Baseline augmentations for YOLO classification training.
    
    YOLO uses a dict-based configuration for augmentations.
    These are standard augmentations similar to default YOLO settings.
    
    Returns:
        dict: Augmentation parameters for YOLO training
    """
    return {
        # Spatial augmentations
        'hsv_h': 0.015,  # HSV-Hue augmentation (fraction)
        'hsv_s': 0.7,    # HSV-Saturation augmentation (fraction)
        'hsv_v': 0.4,    # HSV-Value augmentation (fraction)
        'degrees': 5.0,  # Image rotation (+/- deg)
        'translate': 0.1,  # Image translation (+/- fraction)
        'scale': 0.5,    # Image scale (+/- gain)
        'shear': 0.0,    # Image shear (+/- deg)
        'perspective': 0.0,  # Image perspective (+/- fraction)
        'flipud': 0.0,   # Image flip up-down (probability)
        'fliplr': 0.5,   # Image flip left-right (probability)
        'mosaic': 0.0,   # Mosaic augmentation (not useful for classification)
        'mixup': 0.0,    # MixUp augmentation (probability)
        'copy_paste': 0.0,  # Copy-paste augmentation (probability)
        'erasing': 0.0,  # Random erasing (probability)
    }


def get_yolo_dg_augmentations():
    """
    Domain Generalization (DG) augmentations for YOLO classification.
    
    Heavy augmentations designed to expand the source distribution:
    - Strong HSV (color) augmentations
    - More aggressive spatial transformations
    - Random erasing (cutout)
    
    Returns:
        dict: Augmentation parameters for YOLO training
    """
    return {
        # Strong color augmentations (2-3x stronger than baseline)
        'hsv_h': 0.05,   # Stronger hue variation
        'hsv_s': 0.9,    # Stronger saturation variation
        'hsv_v': 0.7,    # Stronger value/brightness variation
        
        # Aggressive spatial augmentations
        'degrees': 20.0,  # Stronger rotation
        'translate': 0.2,  # More translation
        'scale': 0.8,    # More aggressive scaling
        'shear': 5.0,    # Add shearing
        'perspective': 0.001,  # Slight perspective changes
        
        # Flipping
        'flipud': 0.0,   # No vertical flip (leaves don't typically appear upside down)
        'fliplr': 0.5,   # Horizontal flip
        
        # Additional augmentations
        'mosaic': 0.0,   # Not useful for classification
        'mixup': 0.0,    # Could add if desired, but keep it simple for now
        'copy_paste': 0.0,
        'erasing': 0.3,  # Random erasing (cutout) - helps with generalization
    }


# ============================================================================
# Validation/Test transforms (no augmentation)
# ============================================================================

def get_eval_transforms(img_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Evaluation transforms (no augmentation).
    
    Used for validation and testing.
    
    Args:
        img_size: Target image size
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        torchvision.transforms.Compose
    """
    return T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])


# ============================================================================
# Utility functions
# ============================================================================

def visualize_augmentations(image_path, transforms, n_samples=9):
    """
    Visualize augmentation effects on a sample image.
    
    Args:
        image_path: Path to input image
        transforms: Transform pipeline to apply
        n_samples: Number of augmented samples to generate
    
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Create grid
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(n_samples):
        # Apply transforms
        augmented = transforms(img)
        
        # If tensor, convert back to image for visualization
        if torch.is_tensor(augmented):
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            augmented = augmented * std + mean
            augmented = torch.clamp(augmented, 0, 1)
            augmented = T.ToPILImage()(augmented)
        
        axes[i].imshow(augmented)
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}')
    
    plt.tight_layout()
    return fig


def compare_augmentation_strategies(image_path):
    """
    Compare baseline vs DG augmentations side-by-side.
    
    Args:
        image_path: Path to input image
    
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    
    img = Image.open(image_path).convert('RGB')
    
    baseline_transforms = get_baseline_transforms()
    dg_transforms = get_dg_transforms()
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # Original
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    axes[1, 0].imshow(img)
    axes[1, 0].set_title('Original')
    axes[1, 0].axis('off')
    
    # Baseline augmentations
    for i in range(1, 5):
        augmented = baseline_transforms(img)
        # Denormalize for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        augmented = augmented * std + mean
        augmented = torch.clamp(augmented, 0, 1)
        augmented = T.ToPILImage()(augmented)
        axes[0, i].imshow(augmented)
        axes[0, i].set_title(f'Baseline {i}')
        axes[0, i].axis('off')
    
    # DG augmentations
    for i in range(1, 5):
        augmented = dg_transforms(img)
        # Denormalize for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        augmented = augmented * std + mean
        augmented = torch.clamp(augmented, 0, 1)
        augmented = T.ToPILImage()(augmented)
        axes[1, i].imshow(augmented)
        axes[1, i].set_title(f'DG {i}')
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Baseline', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('DG (Heavy)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig
