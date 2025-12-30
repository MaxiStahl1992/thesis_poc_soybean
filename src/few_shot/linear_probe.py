"""
Linear Probing Utilities
========================

Functions for freezing backbones and fine-tuning classifier heads.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
from tqdm.auto import tqdm
import copy


def freeze_backbone(model: nn.Module, model_type: str = 'resnet') -> nn.Module:
    """
    Freeze all layers except the final classifier head.
    
    Args:
        model: PyTorch model (ResNet, ViT, ConvNeXt, etc.)
        model_type: Type of model ('resnet', 'vit', 'convnext', 'swin')
        
    Returns:
        Model with frozen backbone
        
    Example:
        >>> model = models.resnet50(weights='IMAGENET1K_V2')
        >>> model.fc = nn.Linear(model.fc.in_features, 3)
        >>> model = freeze_backbone(model, model_type='resnet')
        >>> # Only model.fc parameters will have requires_grad=True
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze classifier head based on model type
    if model_type.lower() == 'resnet':
        # ResNet: unfreeze fc layer
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_type.lower() == 'vit':
        # ViT: unfreeze heads.head
        for param in model.heads.head.parameters():
            param.requires_grad = True
    elif model_type.lower() in ['convnext', 'swin']:
        # ConvNeXt/Swin: unfreeze classifier
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Verify freezing
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Frozen backbone: {trainable_params:,} / {total_params:,} params trainable "
          f"({trainable_params/total_params*100:.2f}%)")
    
    return model


def linear_probe_finetune(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    device: str = 'cuda',
    verbose: bool = True,
    early_stopping_patience: int = 10
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Fine-tune only the classifier head (linear probing).
    
    Args:
        model: Model with frozen backbone (use freeze_backbone first)
        train_loader: Training data loader (k-shot subset)
        val_loader: Optional validation loader
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        device: Device to train on
        verbose: Print training progress
        early_stopping_patience: Stop if no improvement for N epochs
        
    Returns:
        Tuple of (fine-tuned model, training history)
        
    Example:
        >>> model = freeze_backbone(model, 'resnet')
        >>> finetuned_model, history = linear_probe_finetune(
        ...     model, k_shot_loader, val_loader, num_epochs=50
        ... )
    """
    model = model.to(device)
    model.train()
    
    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") if verbose else train_loader
        for images, labels in iterator:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, "
                      f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
            
            # Early stopping check
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        else:
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}")
        
        scheduler.step()
    
    # Restore best model if validation was used
    if val_loader is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"\nRestored best model (Val Acc: {best_val_acc:.4f})")
    
    return model, history


def evaluate_linear_probe(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda'
) -> Dict[str, any]:
    """
    Evaluate a linear probe model.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        Dictionary containing predictions, labels, and per-sample logits
        
    Example:
        >>> results = evaluate_linear_probe(model, test_loader)
        >>> print(f"Accuracy: {results['accuracy']:.4f}")
    """
    model = model.to(device)
    model.eval()
    
    all_labels = []
    all_preds = []
    all_logits = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_logits.extend(outputs.cpu().numpy())
    
    # Calculate accuracy
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    
    return {
        'labels': all_labels,
        'predictions': all_preds,
        'logits': all_logits,
        'accuracy': accuracy
    }


def get_trainable_param_count(model: nn.Module) -> Tuple[int, int]:
    """
    Count trainable vs total parameters.
    
    Returns:
        Tuple of (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
