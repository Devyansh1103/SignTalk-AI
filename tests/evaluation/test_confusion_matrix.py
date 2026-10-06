"""
Unit tests for Confusion Matrix computation, top confusions, and plotting.
"""

import pytest
import os
import numpy as np
import tempfile

from src.evaluation.confusion_matrix import (
    compute_confusion_matrix,
    analyze_top_confusions,
    plot_confusion_matrix
)


def test_confusion_matrix_computation():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2])
    cm, cm_norm = compute_confusion_matrix(y_true, y_pred, num_classes=3)

    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1
    assert cm[0, 1] == 1  # 1 misclassification 0 -> 1
    assert cm_norm[0, 0] == 0.5
    assert cm_norm[0, 1] == 0.5
    assert cm_norm[1, 1] == 1.0


def test_top_confusions_analysis():
    cm = np.zeros((3, 3), dtype=np.int64)
    cm[0, 1] = 4  # Class 0 confused with Class 1 4 times
    cm[1, 2] = 2  # Class 1 confused with Class 2 2 times
    class_names = ["hello", "thankyou", "good"]

    top = analyze_top_confusions(cm, class_names, top_k=2)
    assert len(top) == 2
    assert top[0]["true_label"] == "hello"
    assert top[0]["pred_label"] == "thankyou"
    assert top[0]["count"] == 4
    assert "probable_cause" in top[0]


def test_plot_confusion_matrix():
    cm_norm = np.eye(3, dtype=np.float64)
    class_names = ["A", "B", "C"]
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = os.path.join(tmp_dir, "test_cm.png")
        saved = plot_confusion_matrix(cm_norm, class_names, out_path)
        assert os.path.exists(saved)
        assert os.path.getsize(saved) > 1000
