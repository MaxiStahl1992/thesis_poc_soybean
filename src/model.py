import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights, ResNet50_Weights

def get_model(model_name="convnext_tiny", num_classes=3):
    """
    Load pretrained model and replace the head for 'num_classes'.
    Supported: 'convnext_tiny', 'resnet50'
    """
    if model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    print(f"Model '{model_name}' loaded and adapted for {num_classes} classes.")
    return model

def configure_for_tta(model, method='tent'):
    """
    Configure the model for Test-Time Adaptation (TTA).
    - For TENT: Freeze all weights except for the normalization affine parameters (gamma/beta).
    Supports both BatchNorm (ResNet) and LayerNorm (ConvNeXt).
    """
    if method == 'tent':
        # Freeze all parameters
        model.requires_grad_(False)
        
        # Unfreeze normalization affine parameters (gamma/beta)
        unfrozen_count = 0
        for m in model.modules():
            # Check for BatchNorm or LayerNorm
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                m.requires_grad_(True)
                unfrozen_count += 1
        
        if unfrozen_count == 0:
            print("Warning: No normalization layers found to unfreeze for TENT.")
        else:
            print(f"Model configured for TENT: {unfrozen_count} normalization layers unfrozen.")
    else:
        raise ValueError(f"Unsupported TTA method: {method}")
    
    return model

def apply_adabn(model, dataloader, device):
    """
    Adaptive Batch Normalization (AdaBN).
    Re-estimates Batch Normalization statistics for the target domain.
    Note: Only effective for models with BatchNorm layers.
    """
    model.to(device)
    model.train() # Set to train mode to update BN statistics
    
    # Check if model has any BatchNorm layers
    has_bn = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for m in model.modules())
    if not has_bn:
        print("Warning: Model has no BatchNorm layers. AdaBN will be a no-op.")
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            _ = model(inputs)
            
    print("AdaBN: Domain adaptation via BN statistics update completed.")
    return model
