import torch
import torch.nn.functional as F
from ..tta.base import TTAOptimizer

class EataOptimizer(TTAOptimizer):
    """
    EATA: Efficient Anti-forgetting Test-time Adaptation.
    Adds a Fisher-based regularization to prevent forgetting source knowledge.
    Supports sample filtering based on entropy.
    """
    def __init__(self, model, fisher, lr=1e-3, steps=1, e_margin=0.4, d_margin=0.05, fisher_alpha=1.0):
        super().__init__(model, lr)
        self.fisher = fisher
        self.steps = steps
        self.e_margin = e_margin # Lower entropy margin (Filtering redundant samples)
        self.d_margin = d_margin # Upper entropy margin (Filtering noise/outliers)
        self.fisher_alpha = fisher_alpha # Regularization strength
        
        # Store original parameters for regularization
        self.source_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

    def run_adaptation(self, dataloader, device):
        self.check_optimizer()
        self.model.to(device)
        self.model.train()
        
        num_classes = 3 # Healthy, Rust, Frogeye
        high_entropy_cutoff = self.d_margin * torch.log(torch.tensor(num_classes))
        
        for step in range(self.steps):
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                
                with torch.no_grad():
                    outputs = self.model(inputs)
                    probs = F.softmax(outputs, dim=1)
                    entropy = -(probs * torch.log(probs + 1e-5)).sum(dim=1)
                
                # Sample Filtering (EATA core)
                # Filter samples with too much uncertainty (noise)
                mask = entropy < high_entropy_cutoff
                if not mask.any(): continue
                
                inputs = inputs[mask]
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # 1. Entropy Loss
                probs = F.softmax(outputs, dim=1)
                entropy_loss = -(probs * torch.log(probs + 1e-5)).sum(dim=1).mean()
                
                # 2. Anti-forgetting Regularization (Fisher-based)
                reg_loss = 0
                for name, param in self.model.named_parameters():
                    if param.requires_grad and name in self.fisher:
                        reg_loss += (self.fisher[name] * (param - self.source_params[name]).pow(2)).sum()
                
                total_loss = entropy_loss + self.fisher_alpha * reg_loss
                
                total_loss.backward()
                self.optimizer.step()
                
        print(f"EATA: Adaptation completed ({self.steps} steps).")
        return self.model
