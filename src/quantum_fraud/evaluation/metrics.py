"""Imbalance-sensitive metrics; PR-AUC is explicitly average precision."""

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve

from quantum_fraud.preprocessing.pipeline import validate_labels


def checked_scores(labels, scores):
    """Validate aligned finite ranking scores (not necessarily probabilities)."""
    labels = validate_labels(labels)
    scores = np.asarray(scores)
    if scores.ndim != 1 or len(scores) != len(labels) or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite vector aligned with labels")
    return labels, scores


def select_f1_threshold(validation_labels, validation_scores) -> float:
    """Maximize validation F1; choose the highest threshold among exact ties."""
    labels, scores = checked_scores(validation_labels, validation_scores)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denom, out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def select_fpr_threshold(validation_labels, validation_scores, max_fpr: float) -> float:
    """Maximize validation recall subject to FPR; prefer the highest tied threshold."""
    labels, scores = checked_scores(validation_labels, validation_scores)
    if not 0 <= max_fpr <= 1:
        raise ValueError("max_fpr must be in [0, 1]")
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    eligible = np.flatnonzero(fpr <= max_fpr)
    best = eligible[np.argmax(tpr[eligible])]
    threshold = thresholds[best]
    return float(threshold) if np.isfinite(threshold) else float(np.nextafter(scores.max(), np.inf))


def binary_metrics(labels, scores, threshold: float) -> dict:
    """Evaluate one frozen threshold; confusion matrix order is [[TN,FP],[FN,TP]]."""
    labels, scores = checked_scores(labels, scores)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    predictions = scores >= threshold
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc_average_precision": float(average_precision_score(labels, scores)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "false_positive_rate": float(fp / (tn + fp)),
        "false_negative_rate": float(fn / (tp + fn)),
        "confusion_matrix": matrix.tolist(),
    }
