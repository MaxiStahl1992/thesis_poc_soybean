import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import TTAOptimizer


class TENTOptimizer(TTAOptimizer):
    """
    TENT: Test-Time Entropy Minimization.
    
    Method: Minimizes prediction entropy to encourage confident predictions.
    Loss: Entropy with temperature scaling.
    
    Reference: Wang et al. "Tent: Fully Test-Time Adaptation by Entropy Minimization"
    """
    
    def __init__(self, model, device='cpu', lr=1e-4, temperature=1.5, class_names=None, class_weights=None):
        """
        Initialize TENT optimizer.
        
        Args:
            model: PyTorch model configured for TTA
            device: Device to run on
            lr: Learning rate
            temperature: Temperature for softmax scaling
            class_names: Optional list of class names
        """
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr
        )
        super().__init__(model, optimizer, device, class_names)
        self.temperature = temperature
        self.class_weights = class_weights.to(device) if class_weights is not None else None
    
    def entropy_loss(self, outputs):
        """
        Compute entropy loss.
        
        Args:
            outputs: Model logits
            
        Returns:
            loss: Mean entropy across batch
        """
        # Compute softmin entropy
        outputs_t = outputs / self.temperature
        log_probs = F.log_softmax(outputs_t, dim=1)
        probs = F.softmax(outputs_t, dim=1)
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
        Adapt on a single batch by minimizing entropy.
        
        Args:
            batch: Input batch (images or (images, labels))
            
        Returns:
            loss: Entropy loss
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
        
        # Compute entropy loss
        loss = self.entropy_loss(outputs)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss
