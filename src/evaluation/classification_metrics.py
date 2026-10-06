"""
SignTalk AI: Comprehensive Classification Metrics and Evaluation Engine.

Computes:
  - Overall Top-1 & Top-3 Accuracy
  - Macro & Weighted Precision, Recall, and F1-Scores
  - Per-Class Performance Metrics (Precision, Recall, F1, Support)
  - Raw and Normalized Confusion Matrices
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    num_classes: int = 10,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes rigorous classification metrics from ground truth and predictions.
    
    Args:
        y_true: 1D array of true integer class labels [N]
        y_pred: 1D array of predicted integer class labels [N]
        y_probs: Optional 2D array of class posterior probabilities [N, num_classes]
        num_classes: Total number of classes
        class_names: Optional list of class string names
        
    Returns:
        Dictionary of comprehensive evaluation metrics.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    N = len(y_true)

    if N == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "sample_count": 0}

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    # 1. Top-1 Accuracy
    correct = (y_true == y_pred)
    accuracy = float(np.mean(correct))

    # 2. Confusion Matrix [num_classes, num_classes]
    # Rows: True class, Columns: Predicted class
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    # Normalized Confusion Matrix (by true row support)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(np.float64),
        np.maximum(row_sums, 1),
        out=np.zeros_like(cm, dtype=np.float64),
        where=(row_sums > 0)
    )

    # 3. Per-Class Precision, Recall, F1, and Support
    per_class = {}
    precisions = []
    recalls = []
    f1s = []
    supports = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        support = int(cm[c, :].sum())

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        c_name = class_names[c] if c < len(class_names) else f"Class_{c}"
        per_class[c_name] = {
            "class_id": c,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": support
        }

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)

    # 4. Macro Averages (Unweighted average across classes)
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    # 5. Weighted Averages (Weighted by class support)
    total_support = sum(supports)
    if total_support > 0:
        weights = np.array(supports, dtype=np.float64) / total_support
        weighted_precision = float(np.sum(np.array(precisions) * weights))
        weighted_recall = float(np.sum(np.array(recalls) * weights))
        weighted_f1 = float(np.sum(np.array(f1s) * weights))
    else:
        weighted_precision = macro_precision
        weighted_recall = macro_recall
        weighted_f1 = macro_f1

    # 6. Top-3 Accuracy if probabilities available and num_classes >= 3
    top3_accuracy = None
    if y_probs is not None and num_classes >= 3:
        y_probs = np.asarray(y_probs)
        top3_preds = np.argsort(y_probs, axis=-1)[:, -3:]
        top3_correct = [y_true[i] in top3_preds[i] for i in range(N)]
        top3_accuracy = float(np.mean(top3_correct))

    metrics = {
        "sample_count": N,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_precision, 4),
        "weighted_recall": round(weighted_recall, 4),
        "weighted_f1": round(weighted_f1, 4),
        "top3_accuracy": round(top3_accuracy, 4) if top3_accuracy is not None else None,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": np.round(cm_norm, 4).tolist(),
        "per_class": per_class
    }

    return metrics


def format_metrics_table(metrics: Dict[str, Any]) -> str:
    """Formats per-class evaluation metrics into a human-readable ASCII table."""
    lines = []
    lines.append(f"{'Class Name':<16} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    lines.append("-" * 56)

    for cname, vals in metrics["per_class"].items():
        lines.append(
            f"{cname:<16} {vals['precision']:<10.4f} {vals['recall']:<10.4f} "
            f"{vals['f1_score']:<10.4f} {vals['support']:<8}"
        )

    lines.append("-" * 56)
    lines.append(
        f"{'Macro Avg':<16} {metrics['macro_precision']:<10.4f} {metrics['macro_recall']:<10.4f} "
        f"{metrics['macro_f1']:<10.4f} {metrics['sample_count']:<8}"
    )
    lines.append(
        f"{'Weighted Avg':<16} {metrics['weighted_precision']:<10.4f} {metrics['weighted_recall']:<10.4f} "
        f"{metrics['weighted_f1']:<10.4f} {metrics['sample_count']:<8}"
    )
    lines.append(f"{'Overall Acc':<16} {metrics['accuracy']:<10.4f}")

    return "\n".join(lines)
