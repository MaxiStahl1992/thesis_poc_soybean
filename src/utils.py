import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score

def set_seed(seed: int):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")

def calculate_metrics(y_true, y_pred, class_names):
    """
    Calculate Accuracy, Macro F1, and specific class-wise metrics.
    Specifically tracks Rust Recall (Safety) and Frogeye Precision.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Class-wise metrics
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    
    rust_idx = class_names.index('Rust')
    frogeye_idx = class_names.index('Frogeye')
    
    rust_recall = recalls[rust_idx]
    frogeye_precision = precisions[frogeye_idx]
    
    # Track Rust -> Healthy False Negatives (Safety Hazard)
    rust_true_indices = [i for i, label in enumerate(y_true) if label == rust_idx]
    rust_pred_as_healthy = sum(1 for i in rust_true_indices if y_pred[i] == class_names.index('Healthy'))
    
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision_Macro': precision_macro,
        'Recall_Macro': recall_macro,
        'Rust_Recall': rust_recall,
        'Frogeye_Precision': frogeye_precision,
        'Rust_as_Healthy_FN': rust_pred_as_healthy,
        'Confusion_Matrix': cm
    }

def log_experiment(run_id, seed, model_name, train_set, test_set, metrics, adaptation="None", log_dir="results/experiment_logs"):
    """Log detailed experiment results to CSV."""
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, "experiment_registry.csv")
    file_exists = os.path.isfile(file_path)
    
    headers = [
        'Run_ID', 'Seed', 'Model', 'Train_Set', 'Test_Set', 
        'Accuracy', 'F1', 'Recall_Macro', 'Rust_Recall', 'Frogeye_Precision', 
        'Rust_as_Healthy_FN', 'Adaptation'
    ]
    
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        
        writer.writerow([
            run_id,
            seed,
            model_name,
            train_set,
            test_set,
            f"{metrics['Accuracy']:.4f}",
            f"{metrics['F1']:.4f}",
            f"{metrics['Recall_Macro']:.4f}",
            f"{metrics['Rust_Recall']:.4f}",
            f"{metrics['Frogeye_Precision']:.4f}",
            metrics['Rust_as_Healthy_FN'],
            adaptation
        ])
    print(f"Experiment logged to {file_path}")

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, save_path=None):
    """Plots and optionally saves a confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()
    plt.close()

def plot_loss_curves(train_losses, val_f1s, title='Training Progress'):
    """Plots training loss and validation F1 curves with a dual axis."""
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

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title(title)
    fig.tight_layout()
    plt.show()
    plt.close()

def save_splits(dataset, train_indices, val_indices, test_indices, seed, dataset_name, log_dir="results/splits"):
    """Saves the filenames used in each split for tracking."""
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, f"{dataset_name}_seed{seed}_splits.txt")
    
    with open(file_path, 'w') as f:
        for split_name, indices in zip(['TRAIN', 'VAL', 'TEST'], [train_indices, val_indices, test_indices]):
            f.write(f"--- {split_name} ---\n")
            if indices is not None:
                for idx in indices:
                    img_path, _ = dataset.samples[idx]
                    f.write(f"{img_path}\n")
            f.write("\n")
    print(f"Splits saved to {file_path}")

def inspect_samples(model, dataloader, device, class_names, num_samples=3):
    """Displays correct and incorrect predictions for each class."""
    model.eval()
    results = {name: {'correct': [], 'incorrect': []} for name in class_names}
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for i in range(len(inputs)):
                true_name = class_names[labels[i].item()]
                pred_name = class_names[preds[i].item()]
                
                img = inputs[i].cpu().numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                if labels[i] == preds[i]:
                    if len(results[true_name]['correct']) < num_samples:
                        results[true_name]['correct'].append((img, pred_name))
                else:
                    if len(results[true_name]['incorrect']) < num_samples:
                        results[true_name]['incorrect'].append((img, pred_name))
            
            all_full = all(len(d['correct']) >= num_samples and len(d['incorrect']) >= num_samples for d in results.values())
            if all_full: break

    for class_name in class_names:
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        fig.suptitle(f'Sample Inspections: {class_name}', fontsize=16)
        for idx in range(num_samples):
            for i, category in enumerate(['correct', 'incorrect']):
                ax = axes[i, idx]
                if idx < len(results[class_name][category]):
                    img, pred = results[class_name][category][idx]
                    ax.imshow(img)
                    ax.set_title(f"Label: {class_name}\nPred: {pred} ({category.capitalize()})")
                ax.axis('off')
        plt.tight_layout()
        plt.show()

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

def log_training_progress(run_id, epoch, train_loss, val_f1, log_dir="results/training_logs"):
    """Logs epoch-wise training progress."""
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, f"{run_id}_progress.csv")
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Epoch', 'Train_Loss', 'Val_F1'])
        writer.writerow([epoch, f"{train_loss:.4f}", f"{val_f1:.4f}"])
