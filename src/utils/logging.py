import os
import csv
import json
import torch
from datetime import datetime
from pathlib import Path


def log_experiment(experiment_name, metrics, model_name='', dataset='', method='', 
                   save_dir='notebooks/results', notes=''):
    """
    Log experiment results to a CSV registry file.
    
    Args:
        experiment_name: Name of the experiment
        metrics: Dictionary containing evaluation metrics
        model_name: Name of the model architecture
        dataset: Dataset name
        method: TTA method name (or 'baseline')
        save_dir: Directory to save the registry
        notes: Optional notes about the experiment
        
    Returns:
        registry_path: Path to the registry file
    """
    # Create results directory if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    registry_path = save_dir / 'experiment_registry.csv'
    
    # Prepare row data
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row = {
        'timestamp': timestamp,
        'experiment_name': experiment_name,
        'model': model_name,
        'dataset': dataset,
        'method': method,
        'accuracy': f"{metrics.get('accuracy', 0):.4f}",
        'precision': f"{metrics.get('precision', 0):.4f}",
        'recall': f"{metrics.get('recall', 0):.4f}",
        'f1': f"{metrics.get('f1', 0):.4f}",
        'notes': notes
    }
    
    # Write to CSV (append mode)
    file_exists = registry_path.exists()
    
    with open(registry_path, 'a', newline='') as f:
        fieldnames = ['timestamp', 'experiment_name', 'model', 'dataset', 'method',
                     'accuracy', 'precision', 'recall', 'f1', 'notes']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row)
    
    print(f"✅ Logged experiment to {registry_path}")
    return registry_path


def save_checkpoint(model, path, metadata=None, optimizer=None):
    """
    Save model checkpoint with optional metadata.
    
    Args:
        model: PyTorch model
        path: Path to save checkpoint
        metadata: Optional dictionary with experiment metadata
        optimizer: Optional optimizer state to save
        
    Returns:
        path: Path where checkpoint was saved
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    if metadata is not None:
        checkpoint['metadata'] = metadata
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, path)
    print(f"✅ Saved checkpoint to {path}")
    return path


def load_checkpoint(path, model, optimizer=None, device='cpu'):
    """
    Load model checkpoint.
    
    Args:
        path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load model on
        
    Returns:
        metadata: Metadata dictionary (if saved)
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    metadata = checkpoint.get('metadata', {})
    print(f"✅ Loaded checkpoint from {path}")
    return metadata


def save_results_json(results_dict, path):
    """
    Save results dictionary as JSON.
    
    Args:
        results_dict: Dictionary to save
        path: Path to save JSON file
        
    Returns:
        path: Path where JSON was saved
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_dict = convert_to_serializable(results_dict)
    
    with open(path, 'w') as f:
        json.dump(serializable_dict, f, indent=2)
    
    print(f"✅ Saved results to {path}")
    return path


def load_results_json(path):
    """
    Load results from JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        results_dict: Loaded dictionary
    """
    with open(path, 'r') as f:
        results_dict = json.load(f)
    
    print(f"✅ Loaded results from {path}")
    return results_dict


def create_experiment_summary(registry_path='notebooks/results/experiment_registry.csv'):
    """
    Create a summary of all experiments from the registry.
    
    Args:
        registry_path: Path to experiment registry CSV
        
    Returns:
        summary_df: Pandas DataFrame with experiment summary
    """
    import pandas as pd
    
    if not Path(registry_path).exists():
        print(f"⚠️  Registry file not found: {registry_path}")
        return None
    
    df = pd.read_csv(registry_path)
    
    # Group by method and calculate statistics
    summary = df.groupby(['model', 'dataset', 'method']).agg({
        'accuracy': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'f1': ['mean', 'std']
    }).round(4)
    
    return summary
