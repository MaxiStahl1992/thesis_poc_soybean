import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import TTAOptimizer


class EATAOptimizer(TTAOptimizer):
    """
    EATA: Entropy with Anti-forgetting regularization.
    
    Method: Combines entropy minimization with Fisher regularization to prevent catastrophic forgetting.
    Loss: Entropy + lambda * Fisher penalty.
    
    The Fisher Information Matrix penalizes large parameter changes from the source model.
    """
    
    def __init__(self, model, fisher_dict, device='cpu', lr=1e-4, fisher_lambda=0.1, 
                 entropy_threshold=0.4, class_names=None, class_weights=None):
        """
        Initialize EATA optimizer.
        
        Args:
            model: PyTorch model configured for TTA
            fisher_dict: Dictionary of Fisher Information for each parameter
            device: Device to run on
            lr: Learning rate
            fisher_lambda: Weight for Fisher regularization
            entropy_threshold: Only adapt samples with entropy below this threshold (lower = more confident)
            class_names: Optional list of class names
        """
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr
        )
        super().__init__(model, optimizer, device, class_names)
        
        # Move fisher dict to device
        self.fisher_dict = {}
        for name, fisher_value in fisher_dict.items():
            self.fisher_dict[name] = fisher_value.to(device)
        
        self.fisher_lambda = fisher_lambda
        self.entropy_threshold = entropy_threshold
        
        # Store initial parameters for Fisher penalty
        self.initial_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.initial_params[name] = param.data.clone()
        
        # Track sample selection stats
        self.adaptation_stats['selected_samples'] = 0
        self.adaptation_stats['total_samples'] = 0
        
        self.class_weights = class_weights.to(device) if class_weights is not None else None
    
    def entropy_loss(self, outputs):
        """
        Compute entropy loss for each sample.
        
        Args:
            outputs: Model logits
            
        Returns:
            entropy: Entropy per sample (not mean)
        """
        probs = F.softmax(outputs, dim=1)
        log_probs = F.log_softmax(outputs, dim=1)
        entropy = -(probs * log_probs).sum(dim=1)
        return entropy
    
    def fisher_penalty(self):
        """
        Compute Fisher regularization penalty.
        
        Returns:
            penalty: Fisher penalty (sum over all parameters)
        """
        penalty = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher_dict:
                fisher = self.fisher_dict[name].to(self.device)
                initial = self.initial_params[name].to(self.device)
                penalty += (fisher * (param - initial) ** 2).sum()
        return penalty
    
    def adapt_batch(self, batch):
        """
        Adapt on a single batch with sample selection and Fisher regularization.
        
        Args:
            batch: Input batch (images or (images, labels))
            
        Returns:
            loss: Total loss (entropy + Fisher penalty)
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
        
        # Compute entropy for each sample
        entropy = self.entropy_loss(outputs)
        
        # Sample selection: only adapt on low-entropy (confident) samples
        selected_mask = entropy < self.entropy_threshold
        
        # Track statistics
        self.adaptation_stats['total_samples'] += images.size(0)
        self.adaptation_stats['selected_samples'] += selected_mask.sum().item()
        
        # Compute losses
        if selected_mask.sum() > 0:
            entropy_per_sample = entropy[selected_mask]
            
            # Apply class weights if available
            if self.class_weights is not None:
                selected_outputs = outputs[selected_mask]
                class_indices = selected_outputs.argmax(dim=1)
                sample_weights = self.class_weights[class_indices]
                entropy_loss = (entropy_per_sample * sample_weights).mean()
            else:
                entropy_loss = entropy_per_sample.mean()
        else:
            entropy_loss = torch.tensor(0.0, device=self.device)
        
        fisher_loss = self.fisher_penalty()
        
        # Total loss
        loss = entropy_loss + self.fisher_lambda * fisher_loss
        
        # Backward pass
        if loss.item() > 0:
            loss.backward()
            self.optimizer.step()
        
        return loss
    
    def reset_stats(self):
        """Reset adaptation statistics including EATA-specific fields."""
        super().reset_stats()  # Call base class reset
        # Add EATA-specific tracking fields
        self.adaptation_stats['selected_samples'] = 0
        self.adaptation_stats['total_samples'] = 0


def compute_fisher_information(model, dataloader, device='cpu', num_samples=500):
    """
    Compute Fisher Information Matrix for model parameters.
    
    This should be called on the source domain validation set before TTA.
    
    Args:
        model: Trained model
        dataloader: DataLoader for source validation set
        device: Device to run on
        num_samples: Maximum number of samples to use
        
    Returns:
        fisher_dict: Dictionary mapping parameter names to Fisher information
    """
    model.eval()
    fisher_dict = {}
    
    # Initialize Fisher dict with zeros
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param.data)
    
    sample_count = 0
    for batch_idx, batch in enumerate(dataloader):
        if sample_count >= num_samples:
            break
        
        # Extract images and labels
        if isinstance(batch, (tuple, list)):
            images, labels = batch
        else:
            continue  # Skip if no labels
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        
        # Backward pass to get gradients
        model.zero_grad()
        loss.backward()
        
        # Accumulate squared gradients (Fisher = E[grad^2])
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_dict[name] += param.grad.data ** 2
        
        sample_count += images.size(0)
    
    # Normalize by number of samples
    for name in fisher_dict:
        fisher_dict[name] /= sample_count
    
    print(f"✅ Computed Fisher Information from {sample_count} samples")
    return fisher_dict
