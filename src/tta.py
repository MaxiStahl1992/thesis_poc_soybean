import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TTAOptimizer:
    """
    TTAOptimizer handles test-time adaptation for a given model.
    Currently implements TENT (Entropy Minimization).
    """

    def __init__(self, model, lr=1e-3, steps=1):
        """
        Initialize with the model (assumed to be configured via configure_for_tta).
        """
        self.model = model
        self.steps = steps
        
        # Collect parameters that require gradients
        self.params = [p for p in model.parameters() if p.requires_grad]
        if len(self.params) == 0:
            # We don't raise here yet to allow initialization, but we'll check in run_adaptation
            self.optimizer = None
        else:
            self.optimizer = optim.Adam(self.params, lr=lr)

    def run_adaptation(self, dataloader, device):
        """
        Run one or more steps of adaptation on the provided unlabeled data.
        """
        if self.optimizer is None:
            raise RuntimeError(
                "TTAOptimizer found no parameters to optimize. "
                "Ensure you called `configure_for_tta` and that the model has normalization layers (BN or LN)."
            )
            
        self.model.to(device)
        self.model.train() # BN layers must be in train mode to use current batch stats or update affine params
        
        for step in range(self.steps):
            for inputs, _ in dataloader: # Labels are strictly for evaluation, NOT used here
                inputs = inputs.to(device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # Calculate Entropy Loss: H(p) = -sum(p * log(p))
                probs = F.softmax(outputs, dim=1)
                entropy_loss = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
                
                # Safety check for grad_fn
                if entropy_loss.grad_fn is None:
                    raise RuntimeError(
                        "Loss tensor does not have a grad_fn. "
                        "This means the model forward pass didn't involve any trainable parameters."
                    )
                
                entropy_loss.backward()
                self.optimizer.step()
                
        print(f"TTA: Adaptation completed for {self.steps} steps.")
        return self.model

def entropy_minimization(model, dataloader, device, lr=1e-3, steps=1):
    """
    Wrapper function for quick access to entropy minimization (TENT).
    """
    tta = TTAOptimizer(model, lr=lr, steps=steps)
    return tta.run_adaptation(dataloader, device)
