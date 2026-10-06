"""
SignTalk AI: Confidence Calibration, ECE, and Threshold Rejection Analysis.

Computes:
  - Confidence distributions for correct vs incorrect predictions
  - Expected Calibration Error (ECE) across M confidence bins
  - Multi-class Brier Score
  - Acceptance/Rejection metrics across threshold spectrum [0.1, 0.95]
  - Identification of overconfident failures and underconfident successes
"""

from typing import List, Dict, Any, Optional, Tuple
import os
import json
import numpy as np
import pandas as pd


def compute_expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Computes Expected Calibration Error (ECE) and returns per-bin statistics.
    
    Args:
        confidences: 1D array of predicted class max probabilities in [0, 1]
        accuracies: 1D boolean array (1 for correct, 0 for incorrect)
        num_bins: Number of equal-width probability bins
        
    Returns:
        (ece_score, bin_details)
    """
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    N = len(confidences)
    if N == 0:
        return 0.0, []

    ece = 0.0
    bin_details = []

    for m in range(num_bins):
        bin_lower = bin_boundaries[m]
        bin_upper = bin_boundaries[m + 1]

        # Select samples falling into bin [lower, upper)
        if m == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_size = int(np.sum(in_bin))
        if bin_size > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_error = abs(bin_acc - bin_conf)
            ece += (bin_size / N) * bin_error

            bin_details.append({
                "bin_idx": m,
                "bin_range": f"[{bin_lower:.2f}, {bin_upper:.2f}]",
                "sample_count": bin_size,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
                "calibration_gap": round(bin_error, 4)
            })
        else:
            bin_details.append({
                "bin_idx": m,
                "bin_range": f"[{bin_lower:.2f}, {bin_upper:.2f}]",
                "sample_count": 0,
                "accuracy": 0.0,
                "confidence": round((bin_lower + bin_upper) / 2.0, 4),
                "calibration_gap": 0.0
            })

    return round(float(ece), 4), bin_details


def compute_brier_score(y_true: np.ndarray, y_probs: np.ndarray, num_classes: int = 10) -> float:
    """Computes Multi-Class Brier Score."""
    N = len(y_true)
    if N == 0:
        return 0.0

    one_hot = np.zeros((N, num_classes), dtype=np.float64)
    for i, yt in enumerate(y_true):
        if 0 <= yt < num_classes:
            one_hot[i, yt] = 1.0

    brier = np.mean(np.sum((y_probs - one_hot) ** 2, axis=1))
    return round(float(brier), 4)


def analyze_threshold_rejection(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    thresholds: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """
    Evaluates system behavior across confidence rejection thresholds.
    """
    if thresholds is None:
        thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]

    N = len(confidences)
    if N == 0:
        return []

    results = []
    for tau in thresholds:
        accepted = confidences >= tau
        rejected = ~accepted

        num_accepted = int(np.sum(accepted))
        num_rejected = int(np.sum(rejected))

        acc_rate = float(num_accepted / N)
        rej_rate = float(num_rejected / N)

        # Accuracy among accepted samples
        if num_accepted > 0:
            post_rej_acc = float(np.mean(accuracies[accepted]))
        else:
            post_rej_acc = 0.0

        # False acceptances: accepted but incorrect
        false_accepts = int(np.sum(accepted & (~accuracies)))
        false_accept_rate = float(false_accepts / N)

        # False rejections: rejected but was actually correct
        false_rejects = int(np.sum(rejected & accuracies))
        false_reject_rate = float(false_rejects / N)

        results.append({
            "threshold": round(tau, 2),
            "accepted_count": num_accepted,
            "rejected_count": num_rejected,
            "acceptance_rate": round(acc_rate, 4),
            "rejection_rate": round(rej_rate, 4),
            "accepted_accuracy": round(post_rej_acc, 4),
            "false_accept_count": false_accepts,
            "false_accept_rate": round(false_accept_rate, 4),
            "false_reject_count": false_rejects,
            "false_reject_rate": round(false_reject_rate, 4)
        })

    return results


def summarize_confidence_distribution(
    confidences: np.ndarray,
    accuracies: np.ndarray
) -> Dict[str, Any]:
    """Computes summary statistics for correct vs incorrect predictions."""
    correct_confs = confidences[accuracies]
    incorrect_confs = confidences[~accuracies]

    return {
        "overall": {
            "mean": round(float(np.mean(confidences)), 4) if len(confidences) > 0 else 0.0,
            "median": round(float(np.median(confidences)), 4) if len(confidences) > 0 else 0.0,
            "std": round(float(np.std(confidences)), 4) if len(confidences) > 0 else 0.0,
            "min": round(float(np.min(confidences)), 4) if len(confidences) > 0 else 0.0,
            "max": round(float(np.max(confidences)), 4) if len(confidences) > 0 else 0.0,
        },
        "correct_predictions": {
            "count": int(len(correct_confs)),
            "mean": round(float(np.mean(correct_confs)), 4) if len(correct_confs) > 0 else 0.0,
            "median": round(float(np.median(correct_confs)), 4) if len(correct_confs) > 0 else 0.0,
            "min": round(float(np.min(correct_confs)), 4) if len(correct_confs) > 0 else 0.0,
        },
        "incorrect_predictions": {
            "count": int(len(incorrect_confs)),
            "mean": round(float(np.mean(incorrect_confs)), 4) if len(incorrect_confs) > 0 else 0.0,
            "median": round(float(np.median(incorrect_confs)), 4) if len(incorrect_confs) > 0 else 0.0,
            "max": round(float(np.max(incorrect_confs)), 4) if len(incorrect_confs) > 0 else 0.0,
        }
    }
