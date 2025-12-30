"""
LAB Color Space Alignment for Domain Adaptation.

Theory:
- LAB color space separates luminance (L) from chrominance (A, B)
- L: Lightness (0=black, 100=white)
- A: Green-Red axis (negative=green, positive=red)
- B: Blue-Yellow axis (negative=blue, positive=yellow)

Strategy:
- Compute mean/std of A and B channels from source dataset
- For each target image, normalize its A/B channels to match source statistics
- This removes color/illumination bias (e.g., red soil vs brown soil)
- Preserves lesion structure since we don't touch spatial information

Reference:
- "A Color-Based Technique for Measuring Visible Algae on Coral Reefs" (Bryson et al.)
- Color transfer in LAB space is a standard technique in computer vision
"""

import numpy as np
import torch
from PIL import Image
from skimage import color
from pathlib import Path
import pickle


def rgb_to_lab(image):
    """
    Convert RGB image to LAB color space.
    
    Args:
        image: PIL Image or numpy array (H, W, 3) in RGB format
    
    Returns:
        numpy array (H, W, 3) in LAB format
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure RGB format (0-255)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Convert to LAB
    lab = color.rgb2lab(image)
    return lab


def lab_to_rgb(lab_image):
    """
    Convert LAB image to RGB.
    
    Args:
        lab_image: numpy array (H, W, 3) in LAB format
    
    Returns:
        PIL Image in RGB format
    """
    # Convert to RGB (0-1 range)
    rgb = color.lab2rgb(lab_image)
    
    # Convert to 0-255 range
    rgb = (rgb * 255).astype(np.uint8)
    
    return Image.fromarray(rgb)


def compute_lab_statistics(dataset, num_samples=None):
    """
    Compute mean and std of A and B channels from a dataset.
    
    Args:
        dataset: PyTorch dataset or list of image paths
        num_samples: Number of samples to use (None = all)
    
    Returns:
        dict: {'a_mean', 'a_std', 'b_mean', 'b_std'}
    """
    a_values = []
    b_values = []
    
    # Handle different input types
    if hasattr(dataset, '__len__') and hasattr(dataset, '__getitem__'):
        # PyTorch dataset
        indices = range(len(dataset))
        if num_samples is not None:
            indices = np.random.choice(indices, size=min(num_samples, len(dataset)), replace=False)
        
        for idx in indices:
            # Get image (handle both (img, label) and img only)
            item = dataset[idx]
            if isinstance(item, tuple):
                img = item[0]
            else:
                img = item
            
            # Convert tensor to PIL if needed
            if torch.is_tensor(img):
                # Denormalize if normalized
                if img.min() < 0:
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img = img * std + mean
                
                # Convert to PIL
                img = torch.clamp(img, 0, 1)
                img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img)
            
            # Convert to LAB
            lab = rgb_to_lab(img)
            a_values.append(lab[:, :, 1].flatten())
            b_values.append(lab[:, :, 2].flatten())
    
    elif isinstance(dataset, list):
        # List of image paths
        paths = dataset if num_samples is None else np.random.choice(dataset, size=min(num_samples, len(dataset)), replace=False)
        
        for path in paths:
            img = Image.open(path).convert('RGB')
            lab = rgb_to_lab(img)
            a_values.append(lab[:, :, 1].flatten())
            b_values.append(lab[:, :, 2].flatten())
    
    else:
        raise ValueError("Dataset must be a PyTorch dataset or list of image paths")
    
    # Concatenate all values
    a_values = np.concatenate(a_values)
    b_values = np.concatenate(b_values)
    
    # Compute statistics
    stats = {
        'a_mean': float(np.mean(a_values)),
        'a_std': float(np.std(a_values)),
        'b_mean': float(np.mean(b_values)),
        'b_std': float(np.std(b_values))
    }
    
    print(f"LAB Statistics computed from {len(a_values)} pixels:")
    print(f"  A channel: mean={stats['a_mean']:.2f}, std={stats['a_std']:.2f}")
    print(f"  B channel: mean={stats['b_mean']:.2f}, std={stats['b_std']:.2f}")
    
    return stats


def apply_lab_alignment(image, source_stats):
    """
    Align a single image's LAB statistics to match source domain.
    
    Args:
        image: PIL Image or numpy array in RGB format
        source_stats: dict with {'a_mean', 'a_std', 'b_mean', 'b_std'}
    
    Returns:
        PIL Image with aligned color statistics
    """
    # Convert to LAB
    lab = rgb_to_lab(image)
    
    # Extract channels
    l_channel = lab[:, :, 0]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    
    # Compute current statistics
    a_mean = np.mean(a_channel)
    a_std = np.std(a_channel)
    b_mean = np.mean(b_channel)
    b_std = np.std(b_channel)
    
    # Normalize and rescale A channel
    a_normalized = (a_channel - a_mean) / (a_std + 1e-8)
    a_aligned = a_normalized * source_stats['a_std'] + source_stats['a_mean']
    
    # Normalize and rescale B channel
    b_normalized = (b_channel - b_mean) / (b_std + 1e-8)
    b_aligned = b_normalized * source_stats['b_std'] + source_stats['b_mean']
    
    # Reconstruct LAB image
    lab_aligned = np.stack([l_channel, a_aligned, b_aligned], axis=2)
    
    # Convert back to RGB
    rgb_aligned = lab_to_rgb(lab_aligned)
    
    return rgb_aligned


class LABAlignmentTransform:
    """
    PyTorch transform for LAB color space alignment.
    
    Usage:
        # Compute source statistics
        source_stats = compute_lab_statistics(source_dataset)
        
        # Create transform
        transform = LABAlignmentTransform(source_stats)
        
        # Apply to target images
        aligned_img = transform(target_img)
    """
    
    def __init__(self, source_stats):
        """
        Args:
            source_stats: dict with {'a_mean', 'a_std', 'b_mean', 'b_std'}
        """
        self.source_stats = source_stats
    
    def __call__(self, img):
        """
        Apply LAB alignment to an image.
        
        Args:
            img: PIL Image
        
        Returns:
            PIL Image with aligned color statistics
        """
        return apply_lab_alignment(img, self.source_stats)
    
    def save(self, path):
        """Save source statistics to file."""
        with open(path, 'wb') as f:
            pickle.dump(self.source_stats, f)
        print(f"LAB statistics saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load source statistics from file."""
        with open(path, 'rb') as f:
            source_stats = pickle.load(f)
        print(f"LAB statistics loaded from {path}")
        return cls(source_stats)


def visualize_lab_alignment(image_path, source_stats, save_path=None):
    """
    Visualize the effect of LAB alignment on a single image.
    
    Args:
        image_path: Path to input image
        source_stats: dict with LAB statistics
        save_path: Optional path to save comparison figure
    
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Apply alignment
    aligned_img = apply_lab_alignment(img, source_stats)
    
    # Create comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(img)
    axes[0].set_title('Original', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(aligned_img)
    axes[1].set_title('LAB Aligned', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    return fig


def compare_lab_distributions(source_dataset, target_dataset, aligned_dataset, num_samples=500):
    """
    Compare LAB channel distributions before and after alignment.
    
    Args:
        source_dataset: Source domain dataset
        target_dataset: Target domain dataset (before alignment)
        aligned_dataset: Target domain dataset (after alignment)
        num_samples: Number of samples to use for visualization
    
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    # Compute statistics for each dataset
    print("Computing source statistics...")
    source_stats = compute_lab_statistics(source_dataset, num_samples)
    
    print("Computing target statistics...")
    target_stats = compute_lab_statistics(target_dataset, num_samples)
    
    print("Computing aligned statistics...")
    aligned_stats = compute_lab_statistics(aligned_dataset, num_samples)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    datasets = ['Source', 'Target', 'Aligned']
    a_means = [source_stats['a_mean'], target_stats['a_mean'], aligned_stats['a_mean']]
    a_stds = [source_stats['a_std'], target_stats['a_std'], aligned_stats['a_std']]
    b_means = [source_stats['b_mean'], target_stats['b_mean'], aligned_stats['b_mean']]
    b_stds = [source_stats['b_std'], target_stats['b_std'], aligned_stats['b_std']]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    # A channel
    axes[0].bar(x - width/2, a_means, width, label='Mean', alpha=0.8)
    axes[0].bar(x + width/2, a_stds, width, label='Std', alpha=0.8)
    axes[0].set_xlabel('Dataset')
    axes[0].set_ylabel('Value')
    axes[0].set_title('A Channel (Green-Red)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # B channel
    axes[1].bar(x - width/2, b_means, width, label='Mean', alpha=0.8)
    axes[1].bar(x + width/2, b_stds, width, label='Std', alpha=0.8)
    axes[1].set_xlabel('Dataset')
    axes[1].set_ylabel('Value')
    axes[1].set_title('B Channel (Blue-Yellow)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig
