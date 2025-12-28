import torch
from tqdm import tqdm

def compute_fisher_diagonal(model, dataloader, device, num_samples=256):
    """
    Computes the diagonal Fisher Information Matrix for normalization parameters.
    Used by EATA for anti-forgetting.
    """
    model.to(device)
    model.eval()
    
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param)
            
    count = 0
    # Use log-likelihood gradient squared for Fisher estimate
    for inputs, _ in tqdm(dataloader, desc="Computing Fisher"):
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
        # Assuming most likely class as 'pseudo-labels' for log-likelihood
        # or just sum of log-probs for simplicity if we don't have true labels
        _, pseudo_labels = torch.max(outputs, 1)
        
        loss = torch.nn.functional.nll_loss(log_probs, pseudo_labels)
        model.zero_grad()
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.data.pow(2)
        
        count += len(inputs)
        if count >= num_samples:
            break
            
    for name in fisher:
        fisher[name] /= count
        
    print("Fisher Diagonal Matrix computed.")
    return fisher
