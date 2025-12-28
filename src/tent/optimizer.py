import torch
import torch.nn.functional as F
from ..tta.base import TTAOptimizer

class TentOptimizer(TTAOptimizer):
    """
    TENT: Test-time Entropy Minimization.
    Optimizes normalization layers (gamma/beta) to minimize prediction entropy.
    """
    def __init__(self, model, lr=1e-3, steps=1):
        super().__init__(model, lr)
        self.steps = steps

    def run_adaptation(self, dataloader, device):
        self.check_optimizer()
        self.model.to(device)
        self.model.train() # Normalization stats updated or affine params updated
        
        for step in range(self.steps):
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # Entropy Loss
                probs = F.softmax(outputs, dim=1)
                entropy_loss = -(probs * torch.log(probs + 1e-5)).sum(dim=1).mean()
                
                entropy_loss.backward()
                self.optimizer.step()
                
        print(f"TENT: Adaptation completed ({self.steps} steps).")
        return self.model
