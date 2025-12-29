import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import TTAOptimizer


class BNPseudoOptimizer(TTAOptimizer):
    """
    Batch Normalization + Pseudo-Labeling adaptation.
    
    Method: Updates batch norm statistics and adapts on high-confidence pseudo-labels.
    Loss: CrossEntropyLoss on pseudo-labels with confidence > threshold.
    """
    
    def __init__(self, model, device='cpu', lr=1e-3, momentum=0.9, confidence_threshold=0.9, 
                 class_names=None, class_weights=None, confidence_thresholds=None):
        """
        Initialize BN+Pseudo optimizer with class balancing support.
        
        Args:
            model: PyTorch model configured for TTA
            device: Device to run on
            lr: Learning rate
            momentum: SGD momentum
            confidence_threshold: Default confidence threshold (if confidence_thresholds not provided)
            class_names: Optional list of class names
            class_weights: Optional tensor of class weights for balanced loss
            confidence_thresholds: Optional dict/list of per-class confidence thresholds
                                  e.g., {0: 0.95, 1: 0.92, 2: 0.85} or [0.95, 0.92, 0.85]
        """
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            momentum=momentum
        )
        super().__init__(model, optimizer, device, class_names)
        self.confidence_threshold = confidence_threshold
        
        # Class balancing
        self.class_weights = class_weights.to(device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Per-class confidence thresholds
        if confidence_thresholds is not None:
            if isinstance(confidence_thresholds, dict):
                self.confidence_thresholds = confidence_thresholds
            else:  # list or tensor
                self.confidence_thresholds = {i: thresh for i, thresh in enumerate(confidence_thresholds)}
        else:
            self.confidence_thresholds = None
        
        # Track adaptation statistics
        self.adaptation_stats['accepted_samples'] = 0
        self.adaptation_stats['total_samples'] = 0
    
    def adapt_batch(self, batch):
        """
        Adapt on a single batch using pseudo-labels.
        
        Args:
            batch: Input batch (images or (images, labels))
            
        Returns:
            loss: Adaptation loss
        """
        # Extract images from batch
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch
        
        images = images.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(images)
        probs = F.softmax(outputs, dim=1)
        
        # Get pseudo-labels for high-confidence samples
        max_probs, pseudo_labels = torch.max(probs, dim=1)

        # Select samples with confidence above threshold
        if self.confidence_thresholds is not None:
            # Per-class confidence thresholds
            confident_mask = torch.zeros_like(max_probs, dtype=torch.bool)
            for class_idx, threshold in self.confidence_thresholds.items():
                class_mask = (pseudo_labels == class_idx)
                confident_mask |= (class_mask & (max_probs >= threshold))
        else:
            # Single global threshold 
            confident_mask = max_probs > self.confidence_threshold
        
        # Track statistics
        self.adaptation_stats['total_samples'] += images.size(0)
        self.adaptation_stats['accepted_samples'] += confident_mask.sum().item()
        
        # Only adapt on confident samples
        if confident_mask.sum() > 0:
            loss = self.criterion(outputs[confident_mask], pseudo_labels[confident_mask])
            loss.backward()
            self.optimizer.step()
        else:
            loss = torch.tensor(0.0)
        
        return loss
