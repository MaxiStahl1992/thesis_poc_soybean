"""
GradCAM: Gradient-weighted Class Activation Mapping
==================================================

Implementation of GradCAM for visualizing what CNNs are looking at.

Reference:
    Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks 
    via Gradient-based Localization" (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple, List, Dict
import matplotlib.pyplot as plt
from pathlib import Path


class GradCAM:
    """
    GradCAM implementation for PyTorch models.
    
    Generates heatmaps showing which regions of the input the model
    focuses on when making predictions.
    
    Example:
        >>> model = load_model('resnet50.pth')
        >>> gradcam = GradCAM(model, target_layer='layer4')
        >>> heatmap = gradcam.generate(image, target_class=2)
        >>> overlay = gradcam.overlay_heatmap(image, heatmap)
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize GradCAM.
        
        Args:
            model: PyTorch model
            target_layer: Name of layer to visualize (default: last conv layer)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # Find target layer
        if target_layer is None:
            # Auto-detect last convolutional layer
            target_layer = self._find_last_conv_layer()
        
        self.target_layer = target_layer
        
        # Hooks for gradients and activations
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _find_last_conv_layer(self) -> str:
        """
        Find the name of the last convolutional layer.
        
        Returns:
            Layer name
        """
        last_conv = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = name
        
        if last_conv is None:
            # For ResNet-like models, try layer4
            if hasattr(self.model, 'layer4'):
                return 'layer4'
            raise ValueError("Could not find convolutional layer")
        
        return last_conv
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Get the target layer module
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_module = module
                break
        
        if target_module is None:
            raise ValueError(f"Layer '{self.target_layer}' not found in model")
        
        # Register hooks
        target_module.register_forward_hook(forward_hook)
        target_module.register_full_backward_hook(backward_hook)
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate GradCAM heatmap for input.
        
        Args:
            input_tensor: Input tensor (1, C, H, W) or (C, H, W)
            target_class: Target class to visualize (if None, use predicted class)
            
        Returns:
            Heatmap as numpy array (H, W) with values in [0, 1]
        """
        # Ensure batch dimension
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients (weights)
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], device=self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def __call__(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Alias for generate()."""
        return self.generate(input_tensor, target_class)


def generate_gradcam(
    model: nn.Module,
    image: torch.Tensor,
    target_class: Optional[int] = None,
    target_layer: Optional[str] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> np.ndarray:
    """
    Generate GradCAM heatmap (convenience function).
    
    Args:
        model: PyTorch model
        image: Input tensor (1, C, H, W) or (C, H, W)
        target_class: Target class to visualize
        target_layer: Layer to visualize (default: last conv layer)
        device: Device to run on
        
    Returns:
        Heatmap as numpy array (H, W) with values in [0, 1]
        
    Example:
        >>> heatmap = generate_gradcam(model, image, target_class=2)
    """
    gradcam = GradCAM(model, target_layer=target_layer, device=device)
    return gradcam.generate(image, target_class)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on original image.
    
    Args:
        image: Original image as numpy array (H, W, 3) in [0, 255]
        heatmap: Heatmap as numpy array (H, W) in [0, 1]
        alpha: Transparency of heatmap (0=invisible, 1=opaque)
        colormap: OpenCV colormap to use
        
    Returns:
        Overlay image as numpy array (H, W, 3) in [0, 255]
        
    Example:
        >>> overlay = overlay_heatmap(img, heatmap, alpha=0.4)
    """
    # Resize heatmap to match image
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to RGB using colormap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Overlay
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


def visualize_gradcam_comparison(
    image: np.ndarray,
    heatmaps: Dict[str, np.ndarray],
    predictions: Dict[str, Tuple[int, float]],
    class_names: List[str],
    true_label: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Visualize side-by-side GradCAM comparison for multiple models.
    
    Args:
        image: Original image (H, W, 3) in [0, 255]
        heatmaps: Dict of model_name -> heatmap
        predictions: Dict of model_name -> (class_idx, confidence)
        class_names: List of class names
        true_label: Ground truth label (optional)
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
        
    Example:
        >>> fig = visualize_gradcam_comparison(
        ...     image=img,
        ...     heatmaps={'Baseline': heatmap1, 'Few-Shot': heatmap2},
        ...     predictions={'Baseline': (2, 0.85), 'Few-Shot': (3, 0.92)},
        ...     class_names=classes,
        ...     true_label='Rust'
        ... )
    """
    n_models = len(heatmaps)
    fig, axes = plt.subplots(1, n_models + 1, figsize=figsize)
    
    # Original image
    axes[0].imshow(image)
    title = "Original Image"
    if true_label:
        title += f"\nTrue: {true_label}"
    axes[0].set_title(title, fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # GradCAM overlays
    for idx, (model_name, heatmap) in enumerate(heatmaps.items(), start=1):
        overlay = overlay_heatmap(image, heatmap, alpha=0.4)
        axes[idx].imshow(overlay)
        
        # Title with prediction
        pred_class, confidence = predictions[model_name]
        pred_name = class_names[pred_class]
        
        # Color based on correctness
        if true_label:
            is_correct = pred_name == true_label
            color = 'green' if is_correct else 'red'
            title = f"{model_name}\nPred: {pred_name} ({confidence:.1%})"
        else:
            color = 'black'
            title = f"{model_name}\n{pred_name} ({confidence:.1%})"
        
        axes[idx].set_title(title, fontsize=12, fontweight='bold', color=color)
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to: {save_path}")
    
    return fig


def get_confused_samples(
    predictions: Dict[str, np.ndarray],
    labels: np.ndarray,
    class_names: List[str],
    baseline_key: str = 'baseline',
    fewshot_key: str = 'fewshot',
    n_samples: int = 5
) -> List[Tuple[int, str, str, str]]:
    """
    Find samples where baseline fails but few-shot succeeds.
    
    Args:
        predictions: Dict of model_name -> prediction array (N,)
        labels: True labels (N,)
        class_names: List of class names
        baseline_key: Key for baseline model
        fewshot_key: Key for few-shot model
        n_samples: Number of samples to return
        
    Returns:
        List of (index, true_class, baseline_pred, fewshot_pred) tuples
        
    Example:
        >>> confused = get_confused_samples(
        ...     predictions={'baseline': preds1, 'fewshot': preds2},
        ...     labels=y_test,
        ...     class_names=classes
        ... )
    """
    baseline_preds = predictions[baseline_key]
    fewshot_preds = predictions[fewshot_key]
    
    # Find indices where baseline is wrong but few-shot is correct
    baseline_wrong = baseline_preds != labels
    fewshot_correct = fewshot_preds == labels
    confused_indices = np.where(baseline_wrong & fewshot_correct)[0]
    
    # Sample random indices
    if len(confused_indices) > n_samples:
        confused_indices = np.random.choice(confused_indices, n_samples, replace=False)
    
    # Get details
    results = []
    for idx in confused_indices:
        true_class = class_names[labels[idx]]
        baseline_pred = class_names[baseline_preds[idx]]
        fewshot_pred = class_names[fewshot_preds[idx]]
        results.append((idx, true_class, baseline_pred, fewshot_pred))
    
    return results
