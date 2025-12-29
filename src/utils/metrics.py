import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def calculate_metrics(y_true, y_pred, class_names=None):
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels (numpy array or list)
        y_pred: Predicted labels (numpy array or list)
        class_names: Optional list of class names for display
        
    Returns:
        metrics_dict: Dictionary containing accuracy, precision, recall, F1, confusion matrix
    """
    if class_names is None:
        class_names = ['Healthy', 'Rust', 'Frogeye']
    
    # Convert to numpy if needed
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    # Calculate metrics (macro average for multi-class)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'per_class': {
            class_names[i]: {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'f1': f1_per_class[i],
                'support': int(support_per_class[i])
            }
            for i in range(len(class_names))
        }
    }
    
    return metrics


def evaluate_model_full(model, dataloader, device='cpu', class_names=None):
    """
    Perform full evaluation of a model on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        class_names: Optional list of class names
        
    Returns:
        metrics_dict: Dictionary with all evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Handle both (images, labels) and images-only batches
            if isinstance(batch, (tuple, list)):
                images, labels = batch
                all_labels.extend(labels.cpu().numpy())
            else:
                images = batch
            
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculate metrics (only if we have labels)
    if len(all_labels) > 0:
        all_labels = np.array(all_labels)
        metrics = calculate_metrics(all_labels, all_preds, class_names)
        metrics['predictions'] = all_preds
        metrics['probabilities'] = all_probs
        metrics['labels'] = all_labels
    else:
        metrics = {
            'predictions': all_preds,
            'probabilities': all_probs
        }
    
    return metrics


def print_metrics(metrics, title="Evaluation Results"):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary from calculate_metrics()
        title: Title to display
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    
    if 'per_class' in metrics:
        print(f"\n{'Per-Class Metrics:':-^60}")
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"\n{class_name}:")
            print(f"  Precision: {class_metrics['precision']:.4f}")
            print(f"  Recall:    {class_metrics['recall']:.4f}")
            print(f"  F1 Score:  {class_metrics['f1']:.4f}")
            print(f"  Support:   {class_metrics['support']}")
    
    print(f"{'='*60}\n")
