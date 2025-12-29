import torch
import torch.nn as nn
import torchvision.models as models


def get_model(name='convnext_tiny', num_classes=3, device='cpu'):
    """
    Load a pretrained model and modify the classifier head for soybean disease classification.
    
    Supported models:
    - 'convnext_tiny': ConvNeXt Tiny (27.8M params)
    - 'resnet50': ResNet-50 (25.6M params) - recommended for TTA
    - 'resnet34': ResNet-34 (21.8M params)
    - 'vit_b_16': Vision Transformer Base (85.8M params)
    
    Args:
        name: Model name
        num_classes: Number of output classes (default: 3 for Healthy/Rust/Frogeye)
        device: Device to load model on
        
    Returns:
        model: Modified model with custom classifier head
    """
    if 'convnext' in name.lower():
        # Load pretrained ConvNeXt Tiny
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        # Replace classifier head
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        
    elif 'resnet50' in name.lower():
        # Load pretrained ResNet-50 (23.5M params)
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Replace classifier head
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif 'resnet34' in name.lower():
        # Load pretrained ResNet-34 (21.8M params)
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        # Replace classifier head
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif 'vit' in name.lower():
        # Load pretrained ViT Base Patch16
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        # Replace classifier head
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown model name: {name}. Supported: convnext_tiny, resnet50, resnet34, vit_b_16")
    
    return model.to(device)


def configure_for_tta(model, method='tent', verbose=True, unfreeze_last_n_blocks=None):
    """
    Configure model for Test-Time Adaptation by unfreezing:
    1. Normalization layers (BatchNorm, LayerNorm, etc.)
    2. Last N blocks/stages of the backbone (optional, auto-configured by architecture)
    3. Classifier head (final linear layer)
    
    Default strategy:
    - ConvNeXt: Only norm + classifier (last stage is too large at 14M params)
    - ResNet: Norm + last 1 block of layer4 + classifier (~1-3% trainable params)
    - ViT: Norm + last 1 transformer block + classifier (~8% trainable params)
    
    Args:
        model: PyTorch model to configure
        method: TTA method name (for logging)
        verbose: Whether to print configuration details
        unfreeze_last_n_blocks: Number of last blocks/stages to unfreeze
                                None = auto (0 for ConvNeXt, 1 for ResNet, 1 for ViT)
        
    Returns:
        model: Configured model (in-place modification)
    """
    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False
    
    # Auto-configure unfreeze_last_n_blocks based on architecture
    if unfreeze_last_n_blocks is None:
        if hasattr(model, 'features'):
            # ConvNeXt: last stage is 14M params, too large for TTA
            unfreeze_last_n_blocks = 0
        elif hasattr(model, 'layer4'):
            # ResNet: unfreeze last 1 block of layer4 for ~1-3% params
            unfreeze_last_n_blocks = 1
        elif hasattr(model, 'encoder'):
            # ViT: last block is ~7M params, acceptable for TTA
            unfreeze_last_n_blocks = 1
        else:
            unfreeze_last_n_blocks = 0
    
    # Step 1: Unfreeze normalization layers
    unfrozen_norm_modules = 0
    norm_types = (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
    
    for name, module in model.named_modules():
        # Check by type (covers standard norms)
        if isinstance(module, norm_types):
            for param in module.parameters():
                param.requires_grad = True
                unfrozen_norm_modules += 1
        
        # Additional check for 'norm' in name (catches ConvNeXt LayerNorm2d)
        elif 'norm' in name.lower() and hasattr(module, 'weight'):
            # Unfreeze weight and bias if they exist
            for param_name in ['weight', 'bias']:
                param = getattr(module, param_name, None)
                if param is not None and not param.requires_grad:
                    param.requires_grad = True
                    unfrozen_norm_modules += 1
    
    # Step 2: Unfreeze last N blocks/stages of backbone (if requested)
    backbone_params_unfrozen = 0
    
    if unfreeze_last_n_blocks > 0:
        if hasattr(model, 'features'):
            # ConvNeXt case: model.features is a Sequential with 8 stages
            # Stages: [0]=stem, [1-7]=blocks, last stage is features[7]
            # Unfreeze last unfreeze_last_n_blocks stages
            num_stages = len(model.features)
            start_idx = max(0, num_stages - unfreeze_last_n_blocks)
            
            for idx in range(start_idx, num_stages):
                for param in model.features[idx].parameters():
                    if not param.requires_grad:  # Don't double-count norms
                        param.requires_grad = True
                        backbone_params_unfrozen += param.numel()
                        
        elif hasattr(model, 'layer4'):
            # ResNet case: model.layer4 contains Bottleneck blocks
            # ResNet-50: layer4 has 3 blocks (each ~2M params)
            # Unfreeze last N blocks of layer4
            blocks = list(model.layer4.children())
            num_blocks = len(blocks)
            start_idx = max(0, num_blocks - unfreeze_last_n_blocks)
            
            for idx in range(start_idx, num_blocks):
                for param in blocks[idx].parameters():
                    if not param.requires_grad:  # Don't double-count norms
                        param.requires_grad = True
                        backbone_params_unfrozen += param.numel()
                        
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
            # ViT case: model.encoder.layers is a list of transformer blocks
            # Unfreeze last N transformer blocks
            num_layers = len(model.encoder.layers)
            start_idx = max(0, num_layers - unfreeze_last_n_blocks)
            
            for idx in range(start_idx, num_layers):
                for param in model.encoder.layers[idx].parameters():
                    if not param.requires_grad:  # Don't double-count norms
                        param.requires_grad = True
                        backbone_params_unfrozen += param.numel()
    
    # Step 3: Unfreeze classifier head
    classifier_params_unfrozen = 0
    
    if hasattr(model, 'classifier') and len(model.classifier) > 2:
        # ConvNeXt case
        for param in model.classifier[2].parameters():
            if not param.requires_grad:
                param.requires_grad = True
                classifier_params_unfrozen += param.numel()
    elif hasattr(model, 'fc'):
        # ResNet case: model.fc is the classifier
        for param in model.fc.parameters():
            if not param.requires_grad:
                param.requires_grad = True
                classifier_params_unfrozen += param.numel()
    elif hasattr(model, 'heads') and hasattr(model.heads, 'head'):
        # ViT case
        for param in model.heads.head.parameters():
            if not param.requires_grad:
                param.requires_grad = True
                classifier_params_unfrozen += param.numel()
            if not param.requires_grad:
                param.requires_grad = True
                classifier_params_unfrozen += param.numel()
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    if verbose:
        print(f"✅ TTA Configuration ({method}):")
        print(f"   - Unfrozen {unfrozen_norm_modules} normalization modules")
        if unfreeze_last_n_blocks > 0:
            print(f"   - Unfrozen last {unfreeze_last_n_blocks} backbone stage(s) ({backbone_params_unfrozen:,} params)")
        print(f"   - Unfrozen classifier head ({classifier_params_unfrozen:,} params)")
        print(f"   - Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model


def clone_model(model):
    """
    Create a deep copy of the model (useful for running multiple TTA methods from same baseline).
    
    Args:
        model: Source model
        
    Returns:
        cloned_model: Independent copy
    """
    import copy
    return copy.deepcopy(model)


def ensemble_models(models, method='average'):
    """
    Create an ensemble from multiple models by averaging their parameters.
    
    Args:
        models: List of models with identical architectures
        method: Ensemble method ('average' or 'weighted')
        
    Returns:
        ensemble_model: Model with averaged parameters
    """
    if not models:
        raise ValueError("No models provided for ensemble")
    
    ensemble_model = clone_model(models[0])
    
    if method == 'average':
        # Simple parameter averaging
        with torch.no_grad():
            for name, param in ensemble_model.named_parameters():
                param.data = torch.stack([
                    dict(m.named_parameters())[name].data 
                    for m in models
                ]).mean(dim=0)
    
    return ensemble_model
