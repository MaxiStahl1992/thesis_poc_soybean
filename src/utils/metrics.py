import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score

def calculate_metrics(y_true, y_pred, class_names):
    """
    Calculate Accuracy, Macro F1, and balanced class-wise metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Class-wise metrics
    labels = list(range(len(class_names)))
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    precisions = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    
    metrics = {
        'Accuracy': acc,
        'F1': f1,
        'Precision_Macro': precision_macro,
        'Recall_Macro': recall_macro,
        'Confusion_Matrix': cm
    }
    
    for i, name in enumerate(class_names):
        metrics[f'{name}_Recall'] = recalls[i]
        metrics[f'{name}_Precision'] = precisions[i]
    
    # Safety tracking (Rust as Healthy FN)
    rust_idx = class_names.index('Rust')
    healthy_idx = class_names.index('Healthy')
    rust_true_indices = [i for i, label in enumerate(y_true) if label == rust_idx]
    rust_pred_as_healthy = sum(1 for i in rust_true_indices if y_pred[i] == healthy_idx)
    metrics['Rust_as_Healthy_FN'] = rust_pred_as_healthy

    # Safety tracking (Frogeye as Healthy FN)
    frogeye_idx = class_names.index('Frogeye')
    frogeye_true_indices = [i for i, label in enumerate(y_true) if label == frogeye_idx]
    frogeye_pred_as_healthy = sum(1 for i in frogeye_true_indices if y_pred[i] == healthy_idx)
    metrics['Frogeye_as_Healthy_FN'] = frogeye_pred_as_healthy
    
    return metrics
