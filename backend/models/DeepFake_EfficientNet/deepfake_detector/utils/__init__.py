"""Utility functions for training, evaluation, and visualization."""

from deepfake_detector.utils.metrics import (
    calculate_metrics,
    get_EER_states,
    get_HTER_at_thr,
    eval_state
)
from deepfake_detector.utils.logger import setup_logger, get_logger
from deepfake_detector.utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_history
)

__all__ = [
    "calculate_metrics",
    "get_EER_states",
    "get_HTER_at_thr",
    "eval_state",
    "setup_logger",
    "get_logger",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_training_history",
]
