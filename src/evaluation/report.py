"""
SignTalk AI: Comprehensive Report Generator.

Formats:
  - Markdown Evaluation Reports
  - Comparative Performance Tables (Baseline vs ST-GCN vs Transformer)
  - Ablation Summary Tables
  - Robustness Stress Matrices
"""

from typing import Dict, Any, List, Optional
import os
import json
import pandas as pd


def generate_model_summary_markdown(
    model_name: str,
    results: Dict[str, Any],
    output_path: Optional[str] = None
) -> str:
    """Generates a structured Markdown report for a single evaluated model."""
    metrics = results.get("metrics", {})
    bench = results.get("benchmark", {})
    calib = results.get("calibration", {})
    conf_dist = calib.get("confidence_distribution", {})
    top_conf = results.get("top_confusions", [])
    errors = results.get("error_analysis", {})

    md = []
    md.append(f"# SignTalk AI — Evaluation Report: {model_name}")
    md.append("")
    md.append(f"- **Evaluated Checkpoint:** `{results.get('checkpoint', 'N/A')}`")
    md.append(f"- **Evaluation Split:** `{results.get('split', 'test')}`")
    md.append(f"- **Sample Count:** `{results.get('sample_count', 0)}`")
    md.append(f"- **Model Size:** `{results.get('model_size_mb', 'N/A')} MB`")
    md.append(f"- **Total Parameters:** `{bench.get('total_parameters', 'N/A'):,}`")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Primary Classification Performance")
    md.append("")
    md.append("| Metric | Measured Value |")
    md.append("| :--- | :---: |")
    md.append(f"| **Top-1 Accuracy** | **{metrics.get('accuracy', 0.0):.4f}** |")
    md.append(f"| **Macro Precision** | {metrics.get('macro_precision', 0.0):.4f} |")
    md.append(f"| **Macro Recall** | {metrics.get('macro_recall', 0.0):.4f} |")
    md.append(f"| **Macro F1-Score** | **{metrics.get('macro_f1', 0.0):.4f}** |")
    md.append(f"| **Weighted F1-Score** | {metrics.get('weighted_f1', 0.0):.4f} |")
    if metrics.get("top3_accuracy") is not None:
        md.append(f"| **Top-3 Accuracy** | {metrics.get('top3_accuracy', 0.0):.4f} |")
    md.append(f"| **Cross-Entropy Loss** | {metrics.get('loss', 0.0):.4f} |")
    md.append("")
    md.append("## 2. Inference Benchmark & Latency")
    md.append("")
    md.append("| Metric | Value | Budget / Target |")
    md.append("| :--- | :---: | :---: |")
    md.append(f"| **Mean Latency** | `{bench.get('mean_latency_ms', 'N/A')} ms` | $\\le 40.0\\text{{ ms}}$ (25 FPS) |")
    md.append(f"| **Median Latency** | `{bench.get('median_latency_ms', 'N/A')} ms` | - |")
    md.append(f"| **P95 Latency** | `{bench.get('p95_latency_ms', 'N/A')} ms` | $\\le 60.0\\text{{ ms}}$ |")
    md.append(f"| **Throughput** | `{bench.get('throughput_fps', 'N/A')} FPS` | $\\ge 20.0\\text{{ FPS}}$ |")
    md.append(f"| **Meets Real-Time Budget** | `{bench.get('meets_realtime_target', False)}` | Required for Edge |")
    md.append("")
    if calib:
        md.append("## 3. Confidence & Calibration Diagnostics")
        md.append("")
        md.append(f"- **Expected Calibration Error (ECE):** `{calib.get('ece', 'N/A')}`")
        md.append(f"- **Brier Score:** `{calib.get('brier_score', 'N/A')}`")
        if conf_dist.get("correct_predictions"):
            md.append(f"- **Correct Predictions Mean Confidence:** `{conf_dist['correct_predictions'].get('mean', 'N/A')}`")
        if conf_dist.get("incorrect_predictions"):
            md.append(f"- **Incorrect Predictions Mean Confidence:** `{conf_dist['incorrect_predictions'].get('mean', 'N/A')}`")
        md.append("")
    if top_conf:
        md.append("## 4. Top Confused Sign Pairs")
        md.append("")
        md.append("| True Sign | Predicted Sign | Error Count | Probable Contributing Factor |")
        md.append("| :--- | :--- | :---: | :--- |")
        for tc in top_conf:
            md.append(f"| **{tc['true_label']}** | **{tc['pred_label']}** | {tc['count']} | {tc['probable_cause']} |")
        md.append("")
    if errors.get("category_counts"):
        md.append("## 5. Failure Attribution Breakdown")
        md.append("")
        md.append("| Failure Category | Count | Proportion |")
        md.append("| :--- | :---: | :---: |")
        for cat, cnt in errors["category_counts"].items():
            prop = errors["category_proportions"].get(cat, 0.0) * 100.0
            md.append(f"| `{cat}` | {cnt} | {prop:.1f}% |")
        md.append("")

    report_content = "\n".join(md)
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)

    return report_content


def generate_comparison_table(
    baseline_results: Dict[str, Any],
    stgcn_results: Dict[str, Any],
    transformer_results: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Creates structured comparative table between models.
    """
    rows = []

    # Baseline
    b_m = baseline_results.get("metrics", {})
    b_b = baseline_results.get("benchmark", {})
    rows.append({
        "model": "Baseline (BiLSTM)",
        "accuracy": b_m.get("accuracy", None),
        "macro_precision": b_m.get("macro_precision", None),
        "macro_recall": b_m.get("macro_recall", None),
        "macro_f1": b_m.get("macro_f1", None),
        "weighted_f1": b_m.get("weighted_f1", None),
        "latency_ms": b_b.get("mean_latency_ms", None),
        "fps": b_b.get("throughput_fps", None),
        "parameters": b_b.get("total_parameters", None),
        "model_size_mb": baseline_results.get("model_size_mb", None)
    })

    # ST-GCN
    s_m = stgcn_results.get("metrics", {})
    s_b = stgcn_results.get("benchmark", {})
    rows.append({
        "model": "ST-GCN (Graph Conv)",
        "accuracy": s_m.get("accuracy", None),
        "macro_precision": s_m.get("macro_precision", None),
        "macro_recall": s_m.get("macro_recall", None),
        "macro_f1": s_m.get("macro_f1", None),
        "weighted_f1": s_m.get("weighted_f1", None),
        "latency_ms": s_b.get("mean_latency_ms", None),
        "fps": s_b.get("throughput_fps", None),
        "parameters": s_b.get("total_parameters", None),
        "model_size_mb": stgcn_results.get("model_size_mb", None)
    })

    if transformer_results:
        t_m = transformer_results.get("metrics", {})
        t_b = transformer_results.get("benchmark", {})
        rows.append({
            "model": "ST-GCN + Transformer",
            "accuracy": t_m.get("sequence_exact_match", None),
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "weighted_f1": None,
            "latency_ms": t_b.get("mean_latency_ms", None),
            "fps": t_b.get("throughput_fps", None),
            "parameters": t_b.get("total_parameters", None),
            "model_size_mb": transformer_results.get("model_size_mb", None)
        })

    df = pd.DataFrame(rows)
    return df
