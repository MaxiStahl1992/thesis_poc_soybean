import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import TTAOptimizer


class MSUNOptimizer(TTAOptimizer):
    """
    MSUN: Multi-Representation Subdomain Adaptation with Uncertainty Regularization (Simplified for TTA).
    
    Method: Combines subdomain-aware pseudo-labeling with entropy minimization.
    Loss: Cross-entropy on confident pseudo-labels + entropy minimization.
    
    Key differences from other methods:
    - Uses subdomain (class-conditional) confidence thresholds
    - Applies entropy minimization to push decision boundaries to low-density regions
    - Helps prevent class collapse by considering per-class confidence
    
    Reference: Wu et al. "From Laboratory to Field: Unsupervised Domain Adaptation 
               for Plant Disease Recognition in the Wild" (2023)
    """
    
    def __init__(self, model, device='cpu', lr=1e-4, confidence_threshold=0.9,
                 entropy_weight=0.1, class_names=None, class_weights=None,
                 confidence_thresholds=None):
        """
        Initialize MSUN optimizer.
        
        Args:
            model: PyTorch model configured for TTA
            device: Device to run on
            lr: Learning rate
            confidence_threshold: Default confidence threshold for pseudo-labels
            entropy_weight: Weight for entropy minimization term (gamma in paper)
            class_names: Optional list of class names
            class_weights: Optional tensor of class weights for balanced loss
            confidence_thresholds: Optional dict of per-class confidence thresholds
        """
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr
        )
        super().__init__(model, optimizer, device, class_names)
        self.confidence_threshold = confidence_threshold
        self.entropy_weight = entropy_weight
        
        # Class balancing
        self.class_weights = class_weights.to(device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Per-class confidence thresholds (subdomain adaptation)
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
        self.adaptation_stats['entropy_samples'] = 0
    
    def entropy_loss(self, outputs):
        """
        Compute entropy minimization loss.
        
        Entropy minimization pushes decision boundaries to low-density regions,
        making the model more confident on target domain samples.
        
        Args:
            outputs: Model logits
            
        Returns:
            loss: Mean entropy across batch
        """
        probs = F.softmax(outputs, dim=1)
        log_probs = F.log_softmax(outputs, dim=1)
        entropy = -(probs * log_probs).sum(dim=1)
        
        # Apply class weights if available
        if self.class_weights is not None:
            class_indices = probs.argmax(dim=1)
            sample_weights = self.class_weights[class_indices]
            loss = (entropy * sample_weights).mean()
        else:
            loss = entropy.mean()
        
        return loss
    
    def adapt_batch(self, batch):
        """
        Adapt on a single batch using subdomain-aware pseudo-labeling + entropy minimization.
        
        The method:
        1. Generate pseudo-labels for high-confidence samples (subdomain adaptation)
        2. Train on these pseudo-labels (classification loss)
        3. Minimize entropy on all samples (uncertainty regularization)
        
        Args:
            batch: Input batch (images or (images, labels))
            
        Returns:
            loss: Total adaptation loss
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
        
        # Get pseudo-labels for high-confidence samples (subdomain adaptation)
        max_probs, pseudo_labels = torch.max(probs, dim=1)
        
        # Select samples with confidence above threshold (per-class if specified)
        if self.confidence_thresholds is not None:
            # Per-class confidence thresholds (subdomain-aware)
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
        
        # Loss 1: Classification loss on confident pseudo-labels
        if confident_mask.sum() > 0:
            classification_loss = self.criterion(outputs[confident_mask], pseudo_labels[confident_mask])
        else:
            classification_loss = torch.tensor(0.0, device=self.device)
        
        # Loss 2: Entropy minimization on all samples (uncertainty regularization)
        # This pushes decision boundaries to low-density regions
        entropy_regularization = self.entropy_loss(outputs)
        self.adaptation_stats['entropy_samples'] += images.size(0)
        
        # Total loss: classification + entropy minimization
        loss = classification_loss + self.entropy_weight * entropy_regularization
        
        # Backward pass
        if loss.item() > 0:
            loss.backward()
            self.optimizer.step()
        
        return loss
    
    def reset_stats(self):
        """Reset adaptation statistics including MSUN-specific fields."""
        super().reset_stats()
        self.adaptation_stats['accepted_samples'] = 0
        self.adaptation_stats['total_samples'] = 0
        self.adaptation_stats['entropy_samples'] = 0
