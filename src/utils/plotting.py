import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, save_path=None):
    """Plots and optionally saves a confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_loss_curves(train_losses, val_f1s, title='Training Progress'):
    """Plots training loss and validation F1 curves."""
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, train_losses, color=color, label='Train Loss', marker='o')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Val F1', color=color)
    ax2.plot(epochs, val_f1s, color=color, label='Val F1', marker='s')
    ax2.tick_params(axis='y', labelcolor=color)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.title(title)
    fig.tight_layout()
    plt.show()
    plt.close()

def get_specific_errors(model, dataloader, device, class_names, target_class="Frogeye", pred_class="Rust", top_n=5):
    """Finds top N misclassifications for a specific pair."""
    model.eval()
    errors = []
    target_idx = class_names.index(target_class)
    pred_idx = class_names.index(pred_class)
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, 1)
            
            for i in range(len(inputs)):
                if labels[i] == target_idx and preds[i] == pred_idx:
                    img = inputs[i].cpu().numpy().transpose(1, 2, 0)
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img = std * img + mean
                    img = np.clip(img, 0, 1)
                    errors.append((img, confidences[i].item(), f"Conf: {confidences[i].item():.4f}"))
            
    errors.sort(key=lambda x: x[1], reverse=True)
    return errors[:top_n]

def plot_specific_errors(errors, title="Specific Errors"):
    """Plots high-confidence misclassifications."""
    if not errors:
        print(f"No samples found for: {title}")
        return
    n = len(errors)
    fig, axes = plt.subplots(1, n, figsize=(min(n*4, 20), 4))
    if n == 1: axes = [axes]
    fig.suptitle(title, fontsize=16)
    for i, (img, _, info) in enumerate(errors):
        axes[i].imshow(img)
        axes[i].set_title(info)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
