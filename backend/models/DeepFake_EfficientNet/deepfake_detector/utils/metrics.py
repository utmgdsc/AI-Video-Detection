"""
Evaluation metrics for deepfake detection.
Includes EER, ACER, APCER, NPCER, and other forensic metrics.
"""

import numpy as np
import math
from typing import Tuple, List, Dict
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import logging
from typing import Optional, List, Any, Union, Dict

logger = logging.getLogger(__name__)


def eval_state(probs: np.ndarray, labels: np.ndarray, thr: float) -> Tuple[int, int, int, int]:
    """
    Evaluate predictions at a given threshold.

    Args:
        probs: Predicted probabilities for the positive class
        labels: True labels (0 or 1)
        thr: Decision threshold

    Returns:
        Tuple of (TN, FN, FP, TP)
    """
    predict = probs >= thr
    TN = np.sum((labels == 0) & (predict == False))
    FN = np.sum((labels == 1) & (predict == False))
    FP = np.sum((labels == 0) & (predict == True))
    TP = np.sum((labels == 1) & (predict == True))
    return TN, FN, FP, TP


def calculate_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for deepfake detection.

    Args:
        probs: Predicted probabilities for real images (1 = real, 0 = fake)
        labels: True labels (1 = real, 0 = fake)
        threshold: Decision threshold

    Returns:
        Dictionary containing APCER, NPCER, ACER, ACC, and other metrics
    """
    TN, FN, FP, TP = eval_state(probs, labels, threshold)

    # Attack Presentation Classification Error Rate (false fake as real)
    APCER = 1.0 if (FP + TN == 0) else FP / float(FP + TN)

    # Normal Presentation Classification Error Rate (false real as fake)
    # Also known as BPCER (Bona Fide Presentation Classification Error Rate)
    NPCER = 1.0 if (FN + TP == 0) else FN / float(FN + TP)

    # Average Classification Error Rate
    ACER = (APCER + NPCER) / 2.0

    # Accuracy
    ACC = (TP + TN) / (TN + FN + FP + TP) if (TN + FN + FP + TP) > 0 else 0.0

    # Precision and Recall
    PRECISION = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    RECALL = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # F1 Score
    F1 = 2 * (PRECISION * RECALL) / (PRECISION + RECALL) if (PRECISION + RECALL) > 0 else 0.0

    # Specificity
    SPECIFICITY = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    metrics = {
        'accuracy': ACC,
        'apcer': APCER,
        'npcer': NPCER,
        'acer': ACER,
        'precision': PRECISION,
        'recall': RECALL,
        'f1_score': F1,
        'specificity': SPECIFICITY,
        'tp': int(TP),
        'tn': int(TN),
        'fp': int(FP),
        'fn': int(FN)
    }

    return metrics


def get_threshold(probs: np.ndarray, grid_density: int = 10000) -> List[float]:
    """
    Generate threshold values for EER calculation.

    Args:
        probs: Probability array
        grid_density: Number of threshold points to test

    Returns:
        List of threshold values
    """
    thresholds = [i / float(grid_density) for i in range(grid_density + 1)]
    thresholds.append(1.1)
    return thresholds


def get_EER_states(
    probs: np.ndarray,
    labels: np.ndarray,
    grid_density: int = 10000
) -> Tuple[float, float, List[float], List[float]]:
    """
    Calculate Equal Error Rate (EER) and optimal threshold.

    The EER is the point where FAR (False Accept Rate) equals FRR (False Reject Rate).

    Args:
        probs: Predicted probabilities for positive class
        labels: True labels
        grid_density: Number of threshold points to test

    Returns:
        Tuple of (EER, optimal_threshold, FRR_list, FAR_list)
    """
    thresholds = get_threshold(probs, grid_density)
    min_dist = 1.0
    min_dist_states = []
    FRR_list = []
    FAR_list = []

    for thr in thresholds:
        TN, FN, FP, TP = eval_state(probs, labels, thr)

        if (FN + TP == 0):
            FRR = TPR = 1.0
            FAR = FP / float(FP + TN) if (FP + TN) > 0 else 1.0
            TNR = TN / float(TN + FP) if (TN + FP) > 0 else 0.0
        elif (FP + TN == 0):
            TNR = FAR = 1.0
            FRR = FN / float(FN + TP)
            TPR = TP / float(TP + FN)
        else:
            FAR = FP / float(FP + TN)
            FRR = FN / float(FN + TP)
            TNR = TN / float(TN + FP)
            TPR = TP / float(TP + FN)

        dist = math.fabs(FRR - FAR)
        FAR_list.append(FAR)
        FRR_list.append(FRR)

        if dist <= min_dist:
            min_dist = dist
            min_dist_states = [FAR, FRR, thr]

    EER = (min_dist_states[0] + min_dist_states[1]) / 2.0
    optimal_thr = min_dist_states[2]

    return EER, optimal_thr, FRR_list, FAR_list


def get_HTER_at_thr(probs: np.ndarray, labels: np.ndarray, thr: float) -> float:
    """
    Calculate Half Total Error Rate (HTER) at a given threshold.

    HTER is the average of FAR and FRR at a specific threshold.

    Args:
        probs: Predicted probabilities
        labels: True labels
        thr: Decision threshold

    Returns:
        HTER value
    """
    TN, FN, FP, TP = eval_state(probs, labels, thr)

    if (FN + TP == 0):
        FRR = 1.0
        FAR = FP / float(FP + TN) if (FP + TN) > 0 else 1.0
    elif (FP + TN == 0):
        FAR = 1.0
        FRR = FN / float(FN + TP)
    else:
        FAR = FP / float(FP + TN)
        FRR = FN / float(FN + TP)

    HTER = (FAR + FRR) / 2.0
    return HTER


def calculate_comprehensive_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    preds: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate all metrics including EER, ACER, accuracy, etc.

    Args:
        probs: Predicted probabilities for positive class
        labels: True labels
        preds: Predicted class labels (optional, will be computed from probs if not provided)

    Returns:
        Dictionary with all metrics
    """
    if preds is None:
        preds = (probs >= 0.5).astype(int)

    # Standard metrics
    metrics = calculate_metrics(probs, labels, threshold=0.5)

    # EER and optimal threshold
    EER, optimal_thr, _, _ = get_EER_states(probs, labels)
    metrics['eer'] = EER
    metrics['optimal_threshold'] = optimal_thr

    # HTER at threshold 0.5
    HTER = get_HTER_at_thr(probs, labels, 0.5)
    metrics['hter'] = HTER

    # Accuracy at optimal threshold
    optimal_preds = (probs >= optimal_thr).astype(int)
    optimal_acc = accuracy_score(labels, optimal_preds)
    metrics['accuracy_at_optimal_thr'] = optimal_acc

    # AUC-ROC if possible
    try:
        auc = roc_auc_score(labels, probs)
        metrics['auc_roc'] = auc
    except:
        logger.warning("Could not calculate AUC-ROC")
        metrics['auc_roc'] = 0.0

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """
    Pretty print metrics.

    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics display
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")

    for key, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            print(f"{key:.<30} {value:>10d}")
        else:
            print(f"{key:.<30} {value:>10.4f}")

    print(f"{'='*60}\n")
