"""
Visualization utilities for model evaluation and training monitoring.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str] = ['Fake', 'Real'],
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot confusion matrix using seaborn heatmap.

    Args:
        confusion_matrix: 2x2 confusion matrix
        class_names: Names of classes
        normalize: Whether to normalize values
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Example:
        >>> from sklearn.metrics import confusion_matrix
        >>> cm = confusion_matrix(y_true, y_pred)
        >>> plot_confusion_matrix(cm, save_path='confusion_matrix.png')
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=['Predicted ' + cn for cn in class_names],
        yticklabels=class_names,
        linewidths=2,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curve(
    FRR_list: List[float],
    FAR_list: List[float],
    eer: float,
    title: str = 'ROC Curve (FAR vs FRR)',
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot ROC curve showing FAR vs FRR.

    Args:
        FRR_list: False Rejection Rates
        FAR_list: False Acceptance Rates
        eer: Equal Error Rate
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Example:
        >>> EER, thr, FRR_list, FAR_list = get_EER_states(probs, labels)
        >>> plot_roc_curve(FRR_list, FAR_list, EER, save_path='roc_curve.png')
    """
    plt.figure(figsize=(10, 8))

    # Plot ROC curve
    plt.plot(FAR_list, FRR_list, marker='.', label=f'ROC Curve (EER={eer:.4f})', linewidth=2)

    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier', alpha=0.5)

    # Plot EER point
    eer_idx = np.argmin(np.abs(np.array(FAR_list) - np.array(FRR_list)))
    plt.plot(FAR_list[eer_idx], FRR_list[eer_idx], 'ro', markersize=10, label=f'EER Point')

    plt.xlabel('False Acceptance Rate (FAR)', fontsize=12)
    plt.ylabel('False Rejection Rate (FRR)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: List[str] = ['loss', 'accuracy'],
    title: str = 'Training History',
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot training and validation metrics over epochs.

    Args:
        history: Dictionary with training history
                 e.g., {'train_loss': [...], 'val_loss': [...], ...}
        metrics: List of metrics to plot
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Example:
        >>> history = {
        ...     'train_loss': [0.5, 0.4, 0.3],
        ...     'val_loss': [0.6, 0.5, 0.4],
        ...     'train_accuracy': [0.7, 0.8, 0.85],
        ...     'val_accuracy': [0.65, 0.75, 0.8]
        ... }
        >>> plot_training_history(history, save_path='training_history.png')
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(8 * num_metrics, 6))

    if num_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        train_key = f'train_{metric}'
        val_key = f'val_{metric}'

        if train_key in history:
            epochs = range(1, len(history[train_key]) + 1)
            ax.plot(epochs, history[train_key], 'b-o', label=f'Train {metric.capitalize()}', linewidth=2)

        if val_key in history:
            epochs = range(1, len(history[val_key]) + 1)
            ax.plot(epochs, history[val_key], 'r-s', label=f'Val {metric.capitalize()}', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.capitalize()} over Epochs', fontsize=13, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_sample_predictions(
    images: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: List[str] = ['Fake', 'Real'],
    num_samples: int = 16,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot sample images with predictions and ground truth.

    Args:
        images: Array of images (N, H, W, C)
        predictions: Predicted probabilities (N, 2)
        labels: Ground truth labels (N,)
        class_names: Names of classes
        num_samples: Number of samples to display
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Example:
        >>> plot_sample_predictions(
        ...     images, predictions, labels,
        ...     num_samples=9,
        ...     save_path='predictions.png'
        ... )
    """
    num_samples = min(num_samples, len(images))
    grid_size = int(np.ceil(np.sqrt(num_samples)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    for idx in range(num_samples):
        ax = axes[idx]

        # Display image
        img = images[idx]
        if img.shape[0] == 3:  # If channels first
            img = np.transpose(img, (1, 2, 0))

        # Denormalize if needed
        if img.max() <= 1.0:
            img = (img - img.min()) / (img.max() - img.min())

        ax.imshow(img)

        # Get prediction and label
        pred_class = np.argmax(predictions[idx])
        pred_prob = predictions[idx][pred_class]
        true_class = labels[idx]

        # Set title with color based on correctness
        color = 'green' if pred_class == true_class else 'red'
        title = f'Pred: {class_names[pred_class]} ({pred_prob:.2f})\nTrue: {class_names[true_class]}'
        ax.set_title(title, color=color, fontsize=10, fontweight='bold')
        ax.axis('off')

    # Hide extra subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Sample predictions saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
