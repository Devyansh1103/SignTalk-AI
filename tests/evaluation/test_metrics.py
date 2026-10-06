"""
Unit tests for Classification & Translation Metrics evaluation engines.
"""

import pytest
import numpy as np
import torch

from src.evaluation.classification_metrics import compute_classification_metrics, format_metrics_table
from src.evaluation.translation_metrics import (
    compute_token_accuracy,
    compute_sequence_accuracy,
    compute_bleu,
    compute_rouge_l,
    compute_chrf,
    compute_wer
)
from src.evaluation.confidence_analysis import (
    compute_expected_calibration_error,
    compute_brier_score,
    analyze_threshold_rejection
)
from src.evaluation.metrics import compute_metrics


def test_perfect_classification():
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4])
    metrics = compute_classification_metrics(y_true, y_pred, num_classes=5)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["sample_count"] == 5


def test_empty_predictions():
    y_true = np.array([])
    y_pred = np.array([])
    metrics = compute_classification_metrics(y_true, y_pred, num_classes=5)
    assert metrics["sample_count"] == 0
    assert metrics["accuracy"] == 0.0


def test_imbalanced_f1():
    # 4 samples of class 0, 1 sample of class 1
    y_true = np.array([0, 0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0, 0])  # Misses class 1 completely

    metrics = compute_classification_metrics(y_true, y_pred, num_classes=2)
    assert metrics["accuracy"] == 0.80  # 4/5
    assert metrics["macro_f1"] < 0.50
    assert metrics["weighted_f1"] > metrics["macro_f1"]


def test_confusion_matrix_shape():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([1, 1, 2])
    metrics = compute_classification_metrics(y_true, y_pred, num_classes=3)

    cm = np.array(metrics["confusion_matrix"])
    assert cm.shape == (3, 3)
    assert cm.sum() == 3


def test_top3_accuracy():
    y_true = np.array([0, 1])
    probs = np.array([
        [0.2, 0.4, 0.3, 0.1],  # 0 is in top 3
        [0.4, 0.1, 0.3, 0.2]   # 1 is rank 4
    ])
    metrics = compute_classification_metrics(y_true, np.argmax(probs, axis=-1), y_probs=probs, num_classes=4)
    assert metrics["top3_accuracy"] == 0.5


def test_translation_sequence_metrics():
    # Target: [BOS(2), HELLO(4), EOS(3), PAD(0)]
    # Prediction: [BOS(2), HELLO(4), EOS(3), PAD(0)]
    targets = torch.tensor([[2, 4, 3, 0], [2, 5, 3, 0]])
    preds_perfect = torch.tensor([[2, 4, 3, 0], [2, 5, 3, 0]])
    preds_wrong = torch.tensor([[2, 6, 3, 0], [2, 5, 3, 0]])

    assert compute_sequence_accuracy(preds_perfect, targets, pad_idx=0, eos_idx=3) == 1.0
    assert compute_sequence_accuracy(preds_wrong, targets, pad_idx=0, eos_idx=3) == 0.5
    assert compute_token_accuracy(preds_perfect, targets, pad_idx=0) == 1.0


def test_bleu_and_rouge():
    hypotheses = [["hello", "world"]]
    references = [["hello", "world"]]
    bleu = compute_bleu(hypotheses, references)
    rouge = compute_rouge_l(hypotheses, references)

    assert bleu["bleu_1"] == 1.0
    assert bleu["bleu_2"] == 1.0
    assert rouge["rouge_l_f1"] == 1.0


def test_calibration_ece():
    # Perfectly calibrated predictions
    confidences = np.array([0.9, 0.9, 0.1, 0.1])
    accuracies = np.array([True, True, False, False])
    ece, bins = compute_expected_calibration_error(confidences, accuracies, num_bins=10)
    assert ece < 0.20

    brier = compute_brier_score(np.array([0, 1]), np.array([[1.0, 0.0], [0.0, 1.0]]), num_classes=2)
    assert brier == 0.0
