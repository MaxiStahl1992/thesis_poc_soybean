"""
Visualization Utilities for Few-Shot Learning
=============================================

Functions for visualizing Return on Labeling and comparing results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import confusion_matrix


def plot_return_on_labeling(
    k_shots: List[int],
    f1_scores: List[float],
    zero_shot_f1: float,
    baseline_methods: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the "Return on Labeling" curve showing F1 improvement vs number of labeled examples.
    
    Args:
        k_shots: List of k values (e.g., [5, 10, 15])
        f1_scores: List of F1 scores for each k
        zero_shot_f1: Zero-shot baseline F1
        baseline_methods: Optional dict of method_name -> F1 score for comparison
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
        
    Example:
        >>> fig = plot_return_on_labeling(
        ...     k_shots=[5, 10, 15],
        ...     f1_scores=[0.75, 0.80, 0.82],
        ...     zero_shot_f1=0.65,
        ...     baseline_methods={'EATA': 0.68, 'TENT': 0.67}
        ... )
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Convert k to total labeled images (k per class × 3 classes)
    total_images = [k * 3 for k in k_shots]
    
    # Plot 1: F1 Score vs Number of Labeled Images
    ax1.plot([0] + total_images, [zero_shot_f1] + f1_scores, 
             marker='o', linewidth=2, markersize=8, label='Few-Shot Linear Probe', color='#2E86AB')
    ax1.axhline(y=zero_shot_f1, color='gray', linestyle='--', linewidth=1.5, 
                label=f'Zero-Shot Baseline (F1={zero_shot_f1:.3f})', alpha=0.7)
    
    # Add baseline methods if provided
    if baseline_methods:
        colors = ['#A23B72', '#F18F01', '#C73E1D']
        for i, (method, f1) in enumerate(baseline_methods.items()):
            ax1.axhline(y=f1, color=colors[i % len(colors)], linestyle=':', linewidth=1.5,
                       label=f'{method} (F1={f1:.3f})', alpha=0.7)
    
    ax1.set_xlabel('Number of Labeled Target Images', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F1 Score (MH Test Set)', fontsize=12, fontweight='bold')
    ax1.set_title('Return on Labeling: F1 vs Labeled Data', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Add annotations for each point
    for x, y, k in zip(total_images, f1_scores, k_shots):
        improvement = y - zero_shot_f1
        ax1.annotate(f'k={k}\n+{improvement:.3f}', 
                    xy=(x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    # Plot 2: F1 Improvement over Zero-Shot
    improvements = [f1 - zero_shot_f1 for f1 in f1_scores]
    colors_bar = ['#06A77D' if imp > 0.05 else '#F4D35E' if imp > 0.02 else '#EE964B' 
                  for imp in improvements]
    
    bars = ax2.bar(total_images, improvements, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(y=0.05, color='green', linestyle='--', linewidth=1, alpha=0.5, 
                label='Significant Threshold (+0.05)')
    
    ax2.set_xlabel('Number of Labeled Target Images', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1 Improvement over Zero-Shot', fontsize=12, fontweight='bold')
    ax2.set_title('Marginal Gain per Labeled Image', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, imp, k in zip(bars, improvements, k_shots):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'+{imp:.3f}\n({k} per class)',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved Return on Labeling plot to: {save_path}")
    
    return fig


def plot_confusion_comparison(
    confusion_matrices: Dict[str, np.ndarray],
    class_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrices for different k-shot settings side by side.
    
    Args:
        confusion_matrices: Dict mapping method name -> confusion matrix
        class_names: List of class names
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
        
    Example:
        >>> cms = {
        ...     'Zero-Shot': cm_zero,
        ...     'k=5': cm_5,
        ...     'k=10': cm_10,
        ...     'k=15': cm_15
        ... }
        >>> fig = plot_confusion_comparison(cms, ['Healthy', 'Rust', 'Frogeye'])
    """
    n_plots = len(confusion_matrices)
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
    
    if n_plots == 1:
        axes = [axes]
    
    for ax, (method, cm) in zip(axes, confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=class_names, yticklabels=class_names,
                   cbar=True, square=True)
        
        # Calculate accuracy
        accuracy = np.trace(cm) / np.sum(cm)
        
        ax.set_title(f'{method}\nAccuracy: {accuracy:.3f}', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_xlabel('Predicted Label', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved confusion matrix comparison to: {save_path}")
    
    return fig


def plot_rust_frogeye_confusion(
    k_shots: List[int],
    confusion_rates: List[float],
    zero_shot_rate: float,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the "Rust vs Frogeye" confusion rate over different k values.
    
    Args:
        k_shots: List of k values
        confusion_rates: List of confusion rates (% of Rust/Frogeye misclassified)
        zero_shot_rate: Zero-shot confusion rate
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
        
    Example:
        >>> fig = plot_rust_frogeye_confusion(
        ...     k_shots=[5, 10, 15],
        ...     confusion_rates=[0.25, 0.15, 0.12],
        ...     zero_shot_rate=0.45
        ... )
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert k to total images
    total_images = [k * 3 for k in k_shots]
    
    # Plot confusion rate
    ax.plot([0] + total_images, [zero_shot_rate] + confusion_rates,
           marker='o', linewidth=2.5, markersize=10, color='#C73E1D', 
           label='Rust ↔ Frogeye Confusion Rate')
    
    ax.axhline(y=zero_shot_rate, color='gray', linestyle='--', linewidth=1.5,
              label=f'Zero-Shot Rate ({zero_shot_rate:.1%})', alpha=0.7)
    
    # Highlight improvement
    for x, y, k in zip(total_images, confusion_rates, k_shots):
        reduction = zero_shot_rate - y
        ax.annotate(f'k={k}\n-{reduction:.1%}',
                   xy=(x, y), xytext=(5, -15), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))
    
    ax.set_xlabel('Number of Labeled Target Images', fontsize=12, fontweight='bold')
    ax.set_ylabel('Confusion Rate (Rust ↔ Frogeye)', fontsize=12, fontweight='bold')
    ax.set_title('Few-Shot Learning Resolves Rust/Frogeye Confusion', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(zero_shot_rate * 1.2, 0.5)])
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved Rust/Frogeye confusion plot to: {save_path}")
    
    return fig


def plot_training_curves(
    histories: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training curves for different k-shot experiments.
    
    Args:
        histories: Dict mapping k -> training history
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(histories)))
    
    for (k, history), color in zip(histories.items(), colors):
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        ax1.plot(epochs, history['train_loss'], 
                label=f'k={k} (Train)', color=color, linewidth=2)
        if 'val_loss' in history and history['val_loss']:
            ax1.plot(epochs, history['val_loss'],
                    label=f'k={k} (Val)', color=color, linewidth=2, linestyle='--')
        
        # Accuracy curves
        ax2.plot(epochs, history['train_acc'],
                label=f'k={k} (Train)', color=color, linewidth=2)
        if 'val_acc' in history and history['val_acc']:
            ax2.plot(epochs, history['val_acc'],
                    label=f'k={k} (Val)', color=color, linewidth=2, linestyle='--')
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Training Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved training curves to: {save_path}")
    
    return fig
