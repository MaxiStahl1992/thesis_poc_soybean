import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix', figsize=(8, 6), save_path=None):
    """
    Plot a normalized confusion matrix with percentages.
    
    Args:
        cm: Confusion matrix (numpy array)
        class_names: List of class names for axes
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    if class_names is None:
        class_names = ['Healthy', 'Rust', 'Frogeye']
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'}, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc' lists
        save_path: Optional path to save the plot
        
    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_method_comparison(results_dict, metric='f1', title=None, save_path=None):
    """
    Plot bar chart comparing different TTA methods.
    
    Args:
        results_dict: Dictionary mapping method names to metrics dictionaries
        metric: Metric to compare ('f1', 'accuracy', etc.)
        title: Optional custom title
        save_path: Optional path to save the plot
        
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    methods = list(results_dict.keys())
    values = [results_dict[m][metric] for m in methods]
    
    # Create color palette
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(methods)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Formatting
    if title is None:
        title = f'{metric.upper()} Score Comparison Across TTA Methods'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel(f'{metric.upper()} Score', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylim(0, max(values) * 1.15)  # Add headroom for labels
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_f1_gains(baseline_f1, tta_results, save_path=None):
    """
    Plot F1 score gains from baseline to TTA methods.
    
    Args:
        baseline_f1: Baseline F1 score (float)
        tta_results: Dictionary mapping TTA method names to F1 scores
        save_path: Optional path to save the plot
        
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    methods = ['Baseline'] + list(tta_results.keys())
    f1_scores = [baseline_f1] + list(tta_results.values())
    gains = [0] + [f1 - baseline_f1 for f1 in tta_results.values()]
    
    # Color code: baseline gray, positive gains green, negative gains red
    colors = ['gray'] + ['green' if g > 0 else 'red' for g in gains[1:]]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(methods, f1_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add baseline line
    ax.axhline(y=baseline_f1, color='black', linestyle='--', linewidth=2, label=f'Baseline: {baseline_f1:.4f}')
    
    # Add value labels and gain labels
    for i, (bar, f1, gain) in enumerate(zip(bars, f1_scores, gains)):
        height = bar.get_height()
        # F1 score label
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{f1:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        # Gain label (skip baseline)
        if i > 0:
            gain_text = f'+{gain:.4f}' if gain > 0 else f'{gain:.4f}'
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                    gain_text,
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    
    ax.set_title('F1 Score: Baseline vs TTA Methods', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylim(0, max(f1_scores) * 1.15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_confusion_matrix_grid(cms_dict, class_names=None, figsize=(16, 12), save_path=None):
    """
    Plot multiple confusion matrices in a grid layout.
    
    Args:
        cms_dict: Dictionary mapping method names to confusion matrices
        class_names: List of class names
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    if class_names is None:
        class_names = ['Healthy', 'Rust', 'Frogeye']
    
    n_methods = len(cms_dict)
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_methods > 1 else [axes]
    
    for idx, (method, cm) in enumerate(cms_dict.items()):
        ax = axes[idx]
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar=True, ax=ax)
        
        ax.set_title(method, fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=10)
    
    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes
