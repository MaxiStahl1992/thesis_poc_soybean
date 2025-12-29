import torch
from abc import ABC, abstractmethod
from pathlib import Path
import os
from ..utils.metrics import evaluate_model_full

class TTAOptimizer(ABC):
    """
    Abstract base class for Test-Time Adaptation optimizers.
    
    All TTA methods should inherit from this class and implement adapt_batch().
    """
    
    def __init__(self, model, optimizer, device='cpu', class_names=None):
        """
        Initialize TTA optimizer.
        
        Args:
            model: PyTorch model (should already be configured for TTA)
            optimizer: PyTorch optimizer
            device: Device to run adaptation on
            class_names: Optional list of class names
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.class_names = class_names
        self.adaptation_stats = {
            'losses': [],
            'batch_count': 0
        }
    
    @abstractmethod
    def adapt_batch(self, batch):
        """
        Adapt the model on a single batch.
        
        Args:
            batch: Input batch (images tensor or tuple of (images, labels))
            
        Returns:
            loss: Adaptation loss value (scalar)
        """
        raise NotImplementedError("Subclasses must implement adapt_batch()")
    
    def adapt(self, dataloader, verbose=True):
        """
        Perform full adaptation loop over a dataloader.
        
        Args:
            dataloader: DataLoader with test data
            verbose: Whether to print progress
            
        Returns:
            self: For method chaining
        """
        self.model.train()  # Enable batch norm updates
        
        for batch_idx, batch in enumerate(dataloader):
            loss = self.adapt_batch(batch)
            
            self.adaptation_stats['losses'].append(loss.item() if torch.is_tensor(loss) else loss)
            self.adaptation_stats['batch_count'] += 1
            
            if verbose and (batch_idx + 1) % 10 == 0:
                avg_loss = sum(self.adaptation_stats['losses'][-10:]) / min(10, len(self.adaptation_stats['losses']))
                print(f"  Batch {batch_idx + 1}/{len(dataloader)} | Loss: {avg_loss:.4f}")
        
        self.model.eval()  # Switch back to eval mode
        
        if verbose:
            avg_loss = sum(self.adaptation_stats['losses']) / len(self.adaptation_stats['losses'])
            print(f"✅ Adaptation complete | Avg Loss: {avg_loss:.4f}")
        
        return self
    
    def adapt_with_early_stopping(self, train_loader, val_loader, max_epochs=5, patience=2, verbose=True):
        """
        NEW: Adapt with early stopping - WORKS WITH individual method's adapt().

        Args:
            train_loader: Adaptation data (MH train)
            val_loader:   Validation data (MH test)  
            max_epochs:   Maximum adaptation epochs
            patience:     Stop if no improvement for this many epochs
            
        Returns:
            self: Adapted model at best validation F1
        """
        best_f1 = -float('inf')
        patience_counter = 0
        best_state = None

        if verbose:
            print(f"🧠 Early stopping adaptation (patience={patience}, max_epochs={max_epochs})")

        for epoch in range(max_epochs):
            if verbose:
                print(f"\nEpoch {epoch+1}/{max_epochs}")
            
            # Reset stats each epoch
            self.reset_stats()
            
            # CALL INDIVIDUAL METHOD'S adapt() - BN Pseudo, MEMO, etc.
            self.adapt(train_loader, verbose=verbose) 
            
            # Evaluate on validation data
            self.model.eval()
            val_metrics = evaluate_model_full(self.model, val_loader, self.device, self.class_names)
            val_f1 = val_metrics['f1']
            
            if verbose:
                print(f"  Val F1: {val_f1:.4f} | Best: {best_f1:.4f}")
            
            # Check for improvement
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                if verbose:
                    print(f"  💾 New best F1: {best_f1:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"🛑 Early stopping at epoch {epoch+1} (no improvement)")
                break

        # Restore best model state
        if best_state is not None:
            self.model.load_state_dict(best_state)
            if verbose:
                print(f"✅ Restored best model (F1={best_f1:.4f})")
        else:
            if verbose:
                print("⚠️ No improvement found, keeping final model")

        return self
        
    def reset_stats(self):
        """Reset adaptation statistics."""
        self.adaptation_stats = {
            'losses': [],
            'batch_count': 0,
            'total_samples': 0,       
            'accepted_samples': 0
        }
