"""
Unit tests for Model Comparison and Relative Improvement computations.
"""

import pytest
import pandas as pd

from src.evaluation.report import generate_comparison_table


def test_comparison_table_structure():
    baseline_res = {
        "metrics": {"accuracy": 0.4167, "macro_f1": 0.3067, "weighted_f1": 0.3111},
        "benchmark": {"mean_latency_ms": 15.0, "throughput_fps": 66.6, "total_parameters": 895370},
        "model_size_mb": 7.2
    }
    stgcn_res = {
        "metrics": {"accuracy": 0.8333, "macro_f1": 0.7667, "weighted_f1": 0.7778},
        "benchmark": {"mean_latency_ms": 22.0, "throughput_fps": 45.4, "total_parameters": 3116938},
        "model_size_mb": 25.9
    }

    df = generate_comparison_table(baseline_res, stgcn_res)
    assert len(df) == 2
    assert "model" in df.columns
    assert "accuracy" in df.columns
    assert df.loc[df["model"] == "ST-GCN (Graph Conv)", "accuracy"].values[0] == 0.8333
    assert df.loc[df["model"] == "Baseline (BiLSTM)", "accuracy"].values[0] == 0.4167
