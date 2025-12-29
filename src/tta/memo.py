import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import TTAOptimizer


class MEMOOptimizer(TTAOptimizer):
    """
    MEMO (Marginal Entropy Minimization with One test point) with Focal Loss.
    
    Method: Uses focal loss on pseudo-labels to handle class imbalance.
    Loss: Focal loss with alpha=0.25, gamma=2.
    """
    
    def __init__(self, model, device='cpu', lr=1e-4, alpha=0.25, gamma=2.0, class_names=None, class_weights=None):
        """
        Initialize MEMO optimizer.
        
        Args:
            model: PyTorch model configured for TTA
            device: Device to run on
            lr: Learning rate
            alpha: Focal loss alpha parameter (class weighting)
            gamma: Focal loss gamma parameter (focusing parameter)
            class_names: Optional list of class names
        """
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr
        )
        super().__init__(model, optimizer, device, class_names)
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights.to(device) if class_weights is not None else None
    
    def focal_loss(self, outputs, targets=None):
        """
        Compute focal loss.
        
        Args:
            outputs: Model logits
            targets: Optional target labels. If None, uses pseudo-labels from argmax
            
        Returns:
            loss: Focal loss value
        """
        probs = F.softmax(outputs, dim=1)
    
        # Get targets (either provided or pseudo-labels)
        if targets is None:
            targets = probs.argmax(dim=1)
        
        # Convert to one-hot if needed
        if targets.dim() == 1:
            targets_one_hot = F.one_hot(targets, num_classes=probs.size(1)).float()
        else:
            targets_one_hot = targets
        
        # Focal loss computation
        pt = (targets_one_hot * probs).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma
        ce_loss = -(targets_one_hot * torch.log(probs + 1e-8)).sum(dim=1)
        
        # Apply class weights if available
        if self.class_weights is not None:
            if targets.dim() == 1:
                sample_weights = self.class_weights[targets]
            else:
                # For soft labels, weight by predicted class
                sample_weights = self.class_weights[probs.argmax(dim=1)]
            focal_loss = (focal_weight * ce_loss * sample_weights).mean()
        else:
            focal_loss = (focal_weight * ce_loss).mean()
        
        return self.alpha * focal_loss
    
    def adapt_batch(self, batch):
        """
        Adapt on a single batch using focal loss on pseudo-labels.
        
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
        
        # Get pseudo-labels with highest confidence
        probs = F.softmax(outputs, dim=1)
        pseudo_labels = torch.argmax(probs, dim=1)
        
        # Compute focal loss with pseudo-labels
        loss = self.focal_loss(outputs, targets=pseudo_labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss
