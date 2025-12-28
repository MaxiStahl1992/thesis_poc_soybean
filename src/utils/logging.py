import os
import csv
import torch
import numpy as np
import random

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

def log_experiment(run_id, seed, model_name, train_set, test_set, metrics, adaptation="None", log_dir="results/experiment_logs"):
    """Log detailed experiment results to CSV."""
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, "experiment_registry.csv")
    file_exists = os.path.isfile(file_path)
    
    # Dynamically handle headers based on metrics keys
    standard_headers = ['Run_ID', 'Seed', 'Model', 'Train_Set', 'Test_Set', 'Adaptation']
    metric_keys = [k for k in metrics.keys() if k != 'Confusion_Matrix']
    headers = standard_headers + metric_keys
    
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        
        row = [run_id, seed, model_name, train_set, test_set, adaptation]
        for k in metric_keys:
            val = metrics[k]
            row.append(f"{val:.4f}" if isinstance(val, (float, np.float32, np.float64)) else val)
        writer.writerow(row)
    print(f"Experiment logged to {file_path}")

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
