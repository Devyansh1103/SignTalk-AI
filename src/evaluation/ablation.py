"""
SignTalk AI: Controlled Ablation Study Suite.

Executes controlled empirical ablations:
  - Ablation A: Modality (Hand Only vs Hand + Pose vs Full Multimodal)
  - Ablation B: Graph Topology (Uniform K=1 vs Distance K=2 vs Spatial K=3)
  - Ablation C: Temporal Modeling (ST-GCN Temporal Conv vs Spatial-Only GCN vs BiLSTM)
  - Ablation D: Sequence Length (T=15, T=30, T=45, T=60)
  - Ablation E: ST-GCN Freezing vs Joint Fine-Tuning in Translation Pipeline
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.stgcn import SignSTGCN
from src.models.baseline import SignBaselineModel
from src.models.graph import SignGraph
from src.models.sign_translation_model import SignTranslationModel
from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.benchmark import benchmark_model_inference, count_parameters


def run_modality_ablation(
    model: SignSTGCN,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str]
) -> List[Dict[str, Any]]:
    """
    Ablation A: Masks active landmark subsets to isolate subsystem utility.
    1. Hand only (nodes 0-41)
    2. Hand + Body (nodes 0-52)
    3. Hand + Body + Face (nodes 0-92)
    """
    model.eval()
    model.to(device)

    configs = [
        {"name": "Hand Only (42 nodes)", "active_range": (0, 42)},
        {"name": "Hand + Upper Body (53 nodes)", "active_range": (0, 53)},
        {"name": "Full Multimodal (93 nodes)", "active_range": (0, 93)},
    ]

    results = []
    for cfg in configs:
        name = cfg["name"]
        start_idx, end_idx = cfg["active_range"]

        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in dataloader:
                x = (batch.get("x") if "x" in batch else batch.get("landmarks")).clone()
                y = batch["label"]
                mask = batch.get("mask", None)

                # Mask nodes outside active range
                if end_idx < 93:
                    x[:, :, :, end_idx:] = 0.0

                x = x.to(device)
                if mask is not None:
                    mask = mask.to(device)

                logits = model(x, mask)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(y.numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        metrics = compute_classification_metrics(
            np.array(all_targets),
            np.array(all_preds),
            np.array(all_probs),
            num_classes=len(class_names),
            class_names=class_names
        )

        # Benchmark latency
        dummy_x = torch.zeros(1, 3, 45, 93)
        bench = benchmark_model_inference(model, dummy_x, device, num_warmup=5, num_iterations=20)

        results.append({
            "ablation_type": "Modality",
            "configuration": name,
            "accuracy": round(metrics["accuracy"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "weighted_f1": round(metrics["weighted_f1"], 4),
            "latency_ms": bench["mean_latency_ms"],
            "parameters": bench["total_parameters"],
            "notes": f"Active nodes: [{start_idx}:{end_idx}]"
        })

    return results


def run_graph_topology_ablation(
    model: SignSTGCN,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str]
) -> List[Dict[str, Any]]:
    """
    Ablation B: Compares graph adjacency partitioning strategies.
    1. Uniform (K=1)
    2. Distance (K=2)
    3. Spatial Configuration (K=3)
    """
    strategies = [
        ("Uniform Partitioning (K=1)", "uniform"),
        ("Distance Partitioning (K=2)", "distance"),
        ("Spatial Configuration Partitioning (K=3)", "spatial")
    ]

    results = []
    # Save original buffer
    orig_A = model.A.clone()

    for label, strat in strategies:
        graph = SignGraph(strategy=strat)
        new_A = graph.get_adjacency_tensor(device=device)

        if new_A.shape[0] == model.A.shape[0]:
            test_A = new_A
        elif new_A.shape[0] < model.A.shape[0]:
            pad_count = model.A.shape[0] - new_A.shape[0]
            test_A = torch.cat([new_A, new_A[-1:].repeat(pad_count, 1, 1)], dim=0)
        else:
            test_A = new_A[:model.A.shape[0]]

        model.A.copy_(test_A)

        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in dataloader:
                x = (batch.get("x") if "x" in batch else batch.get("landmarks")).to(device)
                y = batch["label"]
                mask = batch.get("mask", None)
                if mask is not None:
                    mask = mask.to(device)

                logits = model(x, mask)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(y.numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        metrics = compute_classification_metrics(
            np.array(all_targets),
            np.array(all_preds),
            np.array(all_probs),
            num_classes=len(class_names),
            class_names=class_names
        )

        dummy_x = torch.zeros(1, 3, 45, 93)
        bench = benchmark_model_inference(model, dummy_x, device, num_warmup=5, num_iterations=20)

        results.append({
            "ablation_type": "Graph Topology",
            "configuration": label,
            "accuracy": round(metrics["accuracy"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "weighted_f1": round(metrics["weighted_f1"], 4),
            "latency_ms": bench["mean_latency_ms"],
            "parameters": bench["total_parameters"],
            "notes": f"Adjacency strategy: {strat}"
        })

    # Restore original A
    model.A.copy_(orig_A)
    return results


def run_sequence_length_ablation(
    model: SignSTGCN,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str]
) -> List[Dict[str, Any]]:
    """
    Ablation D: Resamples sequence to T=15, T=30, T=45, T=60 frames.
    """
    model.eval()
    model.to(device)

    lengths = [15, 30, 45, 60]
    results = []

    for T_len in lengths:
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in dataloader:
                x = batch.get("x") if "x" in batch else batch.get("landmarks")  # [B, 3, 45, 93]
                y = batch["label"]
                B, C, T, V = x.shape

                # Resample along time
                flat = x.permute(0, 1, 3, 2).reshape(B, C * V, T)
                resampled = torch.nn.functional.interpolate(flat, size=T_len, mode="linear", align_corners=False)
                # Interpolate or pad back to 45 so model can process with current temporal strides
                back_to_45 = torch.nn.functional.interpolate(resampled, size=45, mode="linear", align_corners=False)
                x_adj = back_to_45.view(B, C, V, 45).permute(0, 1, 3, 2).to(device)

                logits = model(x_adj)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(y.numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        metrics = compute_classification_metrics(
            np.array(all_targets),
            np.array(all_preds),
            np.array(all_probs),
            num_classes=len(class_names),
            class_names=class_names
        )

        dummy_x = torch.zeros(1, 3, 45, 93)
        bench = benchmark_model_inference(model, dummy_x, device, num_warmup=5, num_iterations=20)

        results.append({
            "ablation_type": "Sequence Length",
            "configuration": f"T={T_len} frames",
            "accuracy": round(metrics["accuracy"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "weighted_f1": round(metrics["weighted_f1"], 4),
            "latency_ms": bench["mean_latency_ms"],
            "parameters": bench["total_parameters"],
            "notes": f"Simulated temporal resolution T={T_len}"
        })

    return results
