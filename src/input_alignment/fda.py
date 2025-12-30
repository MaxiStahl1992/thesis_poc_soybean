"""
Fourier Domain Adaptation (FDA) for Domain Adaptation.

Theory:
- Images can be decomposed into frequency components via Fast Fourier Transform (FFT)
- Low frequencies encode global structure (illumination, color tone)
- High frequencies encode local details (edges, textures, lesions)
- Phase information encodes spatial structure
- Amplitude information encodes style/appearance

Strategy:
- Take FFT of target image and source image
- Replace low-frequency amplitude of target with source amplitude
- Keep target's phase intact (preserves lesion structure)
- Inverse FFT to get aligned image

Result:
- Target image keeps its disease patterns (high-freq + phase)
- But adopts source domain's illumination/color (low-freq amplitude)

Reference:
- "FDA: Fourier Domain Adaptation for Semantic Segmentation" (Yang & Soatto, CVPR 2020)
- "Phase Consistent Ecological Domain Adaptation" (Yang et al., CVPR 2020)
"""

import numpy as np
import torch
from PIL import Image
import random


def fourier_domain_adaptation(target_image, source_image, beta=0.01):
    """
    Apply Fourier Domain Adaptation to align target image to source domain.
    
    Args:
        target_image: PIL Image or numpy array (H, W, 3) - image to adapt
        source_image: PIL Image or numpy array (H, W, 3) - style reference
        beta: float in [0, 1] - proportion of spectrum to replace
              0.01 = replace center 1% (low frequencies only)
              0.1 = replace center 10% (more aggressive)
    
    Returns:
        PIL Image - target with source's low-frequency style
    """
    # Convert to numpy arrays
    if isinstance(target_image, Image.Image):
        target_np = np.array(target_image).astype(np.float32)
    else:
        target_np = target_image.astype(np.float32)
    
    if isinstance(source_image, Image.Image):
        source_np = np.array(source_image).astype(np.float32)
    else:
        source_np = source_image.astype(np.float32)
    
    # Ensure same size (resize source if needed)
    if target_np.shape != source_np.shape:
        source_pil = Image.fromarray(source_np.astype(np.uint8))
        source_pil = source_pil.resize((target_np.shape[1], target_np.shape[0]), Image.BILINEAR)
        source_np = np.array(source_pil).astype(np.float32)
    
    # Apply FDA per channel
    adapted_np = np.zeros_like(target_np)
    
    for c in range(3):  # RGB channels
        # Forward FFT (2D)
        target_fft = np.fft.fft2(target_np[:, :, c])
        source_fft = np.fft.fft2(source_np[:, :, c])
        
        # Shift zero frequency to center
        target_fft_shifted = np.fft.fftshift(target_fft)
        source_fft_shifted = np.fft.fftshift(source_fft)
        
        # Extract amplitude and phase
        target_amp = np.abs(target_fft_shifted)
        target_phase = np.angle(target_fft_shifted)
        source_amp = np.abs(source_fft_shifted)
        
        # Get image dimensions
        h, w = target_np.shape[:2]
        
        # Define center region to replace (low frequencies)
        center_h, center_w = h // 2, w // 2
        mask_h = int(h * beta)
        mask_w = int(w * beta)
        
        # Create a copy of target amplitude
        adapted_amp = target_amp.copy()
        
        # Replace center region (low frequencies) with source amplitude
        h1 = max(0, center_h - mask_h // 2)
        h2 = min(h, center_h + mask_h // 2)
        w1 = max(0, center_w - mask_w // 2)
        w2 = min(w, center_w + mask_w // 2)
        
        adapted_amp[h1:h2, w1:w2] = source_amp[h1:h2, w1:w2]
        
        # Reconstruct FFT with adapted amplitude and original phase
        adapted_fft_shifted = adapted_amp * np.exp(1j * target_phase)
        
        # Shift back and inverse FFT
        adapted_fft = np.fft.ifftshift(adapted_fft_shifted)
        adapted_channel = np.fft.ifft2(adapted_fft)
        
        # Take real part and store
        adapted_np[:, :, c] = np.real(adapted_channel)
    
    # Clip to valid range and convert to uint8
    adapted_np = np.clip(adapted_np, 0, 255).astype(np.uint8)
    
    return Image.fromarray(adapted_np)


def apply_fda(target_image, source_images, beta=0.01):
    """
    Apply FDA using a random source image from a list.
    
    This is useful when you have multiple source images and want
    to randomize which one is used for style transfer.
    
    Args:
        target_image: PIL Image to adapt
        source_images: list of PIL Images or single PIL Image
        beta: float - proportion of spectrum to replace
    
    Returns:
        PIL Image - adapted target
    """
    # Handle single image or list
    if isinstance(source_images, Image.Image):
        source_image = source_images
    else:
        source_image = random.choice(source_images)
    
    return fourier_domain_adaptation(target_image, source_image, beta=beta)


class FDATransform:
    """
    PyTorch transform for Fourier Domain Adaptation.
    
    Usage:
        # Load source images
        source_images = [Image.open(p) for p in source_paths]
        
        # Create transform
        fda_transform = FDATransform(source_images, beta=0.01)
        
        # Apply to target images (randomly picks a source each time)
        adapted_img = fda_transform(target_img)
    """
    
    def __init__(self, source_images, beta=0.01):
        """
        Args:
            source_images: list of PIL Images or single PIL Image
            beta: float in [0, 1] - proportion of spectrum to replace
        """
        if isinstance(source_images, Image.Image):
            self.source_images = [source_images]
        else:
            self.source_images = list(source_images)
        
        self.beta = beta
        print(f"FDATransform initialized with {len(self.source_images)} source images (beta={beta})")
    
    def __call__(self, img):
        """
        Apply FDA to an image.
        
        Args:
            img: PIL Image
        
        Returns:
            PIL Image with adapted style
        """
        # Randomly select a source image
        source_img = random.choice(self.source_images)
        
        return fourier_domain_adaptation(img, source_img, beta=self.beta)


def visualize_fda(target_path, source_path, betas=[0.01, 0.05, 0.1], save_path=None):
    """
    Visualize FDA with different beta values.
    
    Args:
        target_path: Path to target image
        source_path: Path to source image
        betas: list of beta values to test
        save_path: Optional path to save figure
    
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    # Load images
    target_img = Image.open(target_path).convert('RGB')
    source_img = Image.open(source_path).convert('RGB')
    
    # Create figure
    fig, axes = plt.subplots(2, len(betas) + 1, figsize=(4 * (len(betas) + 1), 8))
    
    # Original images
    axes[0, 0].imshow(target_img)
    axes[0, 0].set_title('Target (Original)', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(source_img)
    axes[1, 0].set_title('Source (Reference)', fontsize=12)
    axes[1, 0].axis('off')
    
    # Apply FDA with different betas
    for i, beta in enumerate(betas):
        adapted_img = fourier_domain_adaptation(target_img, source_img, beta=beta)
        
        axes[0, i + 1].imshow(adapted_img)
        axes[0, i + 1].set_title(f'FDA (β={beta})', fontsize=12)
        axes[0, i + 1].axis('off')
        
        # Empty space in second row
        axes[1, i + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    return fig


def visualize_frequency_spectrum(image_path, save_path=None):
    """
    Visualize the frequency spectrum of an image.
    
    Helps understand what low/high frequencies represent.
    
    Args:
        image_path: Path to image
        save_path: Optional path to save figure
    
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img).astype(np.float32)
    
    # Use green channel for visualization (middle of RGB)
    channel = img_np[:, :, 1]
    
    # FFT
    fft = np.fft.fft2(channel)
    fft_shifted = np.fft.fftshift(fft)
    
    # Magnitude spectrum (log scale for visibility)
    magnitude = np.abs(fft_shifted)
    magnitude_log = np.log(magnitude + 1)  # +1 to avoid log(0)
    
    # Phase spectrum
    phase = np.angle(fft_shifted)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    im1 = axes[1].imshow(magnitude_log, cmap='hot')
    axes[1].set_title('Amplitude Spectrum (log)', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    im2 = axes[2].imshow(phase, cmap='hsv')
    axes[2].set_title('Phase Spectrum', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    fig.suptitle('FFT Decomposition: Center = Low Freq (style), Edges = High Freq (details)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    return fig


def compare_fda_lab(target_image, source_image, lab_stats, beta=0.01, save_path=None):
    """
    Compare FDA and LAB alignment side by side.
    
    Args:
        target_image: PIL Image or path to target
        source_image: PIL Image or path to source (for FDA)
        lab_stats: dict with LAB statistics (for LAB alignment)
        beta: float for FDA
        save_path: Optional path to save figure
    
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    from .lab_alignment import apply_lab_alignment
    
    # Load images if paths
    if isinstance(target_image, str):
        target_image = Image.open(target_image).convert('RGB')
    if isinstance(source_image, str):
        source_image = Image.open(source_image).convert('RGB')
    
    # Apply transformations
    fda_aligned = fourier_domain_adaptation(target_image, source_image, beta=beta)
    lab_aligned = apply_lab_alignment(target_image, lab_stats)
    
    # Create comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(target_image)
    axes[0, 0].set_title('Target (Original)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(source_image)
    axes[0, 1].set_title('Source (Reference)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(lab_aligned)
    axes[1, 0].set_title('LAB Aligned\n(Color statistics matching)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(fda_aligned)
    axes[1, 1].set_title(f'FDA (β={beta})\n(Frequency domain transfer)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison saved to {save_path}")
    
    return fig
