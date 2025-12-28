import torch
import torch.nn as nn
import torch.optim as optim

class TTAOptimizer:
    """
    Base class for Test-Time Adaptation Optimizers.
    """
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.params = [p for p in model.parameters() if p.requires_grad]
        if len(self.params) == 0:
            self.optimizer = None
        else:
            self.optimizer = optim.Adam(self.params, lr=lr)

    def check_optimizer(self):
        if self.optimizer is None:
            raise RuntimeError(
                "TTAOptimizer found no parameters to optimize. "
                "Ensure parameters have requires_grad=True."
            )
