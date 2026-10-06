"""
Unit tests for Prediction Schema, Error Analyzer, and Confidence Analysis.
"""

import pytest
import pandas as pd
import numpy as np

from src.evaluation.error_analysis import ErrorAnalyzer
from src.evaluation.confidence_analysis import analyze_threshold_rejection, summarize_confidence_distribution


def test_error_analyzer_categorization():
    analyzer = ErrorAnalyzer(class_names=["hello", "thankyou"])

    # Test detection failure trigger
    res1 = analyzer.categorize_error(
        true_label="hello",
        pred_label="thankyou",
        metadata={"valid_frames": 4, "quality_status": "REJECT", "left_hand_detection_rate": 0.0, "right_hand_detection_rate": 0.0},
        confidence=0.6
    )
    assert res1["primary_category"] == "VISUAL_DETECTION_FAILURE"

    # Test landmark noise trigger
    res2 = analyzer.categorize_error(
        true_label="hello",
        pred_label="thankyou",
        metadata={"valid_frames": 25, "missing_landmark_ratio": 0.85, "quality_score": 0.30},
        confidence=0.7
    )
    assert res2["primary_category"] == "REPRESENTATION_LANDMARK_NOISE"

    # Test kinematic similarity fallback
    res3 = analyzer.categorize_error(
        true_label="hello",
        pred_label="thankyou",
        metadata={"valid_frames": 30, "missing_landmark_ratio": 0.55, "quality_score": 0.70},
        confidence=0.65
    )
    assert res3["primary_category"] == "SEMANTIC_KINEMATIC_SIMILARITY"


def test_confidence_distribution_summary():
    confs = np.array([0.9, 0.8, 0.4, 0.5])
    accs = np.array([True, True, False, False])
    summary = summarize_confidence_distribution(confs, accs)

    assert summary["overall"]["mean"] == 0.65
    assert summary["correct_predictions"]["mean"] == 0.85
    assert summary["incorrect_predictions"]["mean"] == 0.45


def test_threshold_rejection_sweep():
    confs = np.array([0.95, 0.85, 0.65, 0.45])
    accs = np.array([True, True, True, False])
    sweep = analyze_threshold_rejection(confs, accs, thresholds=[0.70])

    assert len(sweep) == 1
    # Samples with conf >= 0.70 are 0.95 and 0.85 (both correct)
    assert sweep[0]["accepted_count"] == 2
    assert sweep[0]["rejected_count"] == 2
    assert sweep[0]["accepted_accuracy"] == 1.0
    assert sweep[0]["false_accept_count"] == 0
