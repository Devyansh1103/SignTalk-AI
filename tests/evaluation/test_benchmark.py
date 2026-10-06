"""
Unit tests for Latency and Resource Benchmarking.
"""

import pytest
import torch
import torch.nn as nn

from src.evaluation.benchmark import (
    get_hardware_environment,
    count_parameters,
    benchmark_model_inference
)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x, mask=None):
        return self.fc(x)


def test_hardware_environment_inspection():
    env = get_hardware_environment()
    assert "os" in env
    assert "processor" in env
    assert "pytorch_version" in env
    assert "cuda_available" in env


def test_parameter_counting():
    model = DummyModel()
    total, trainable = count_parameters(model)
    # 10*5 weights + 5 biases = 55
    assert total == 55
    assert trainable == 55


def test_benchmark_inference_timing():
    model = DummyModel()
    dummy_x = torch.randn(2, 10)
    bench = benchmark_model_inference(
        model=model,
        sample_input=dummy_x,
        device=torch.device("cpu"),
        num_warmup=2,
        num_iterations=10,
        target_fps=25.0
    )

    assert bench["num_iterations"] == 10
    assert bench["mean_latency_ms"] > 0
    assert bench["median_latency_ms"] > 0
    assert bench["throughput_fps"] > 0
    assert "meets_realtime_target" in bench
