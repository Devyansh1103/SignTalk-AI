#!/usr/bin/env python3
"""
SignTalk AI: Baseline vs ST-GCN Controlled Comparison CLI.

Loads evaluation metrics from Baseline and ST-GCN runs, calculates absolute
and relative improvements across all dimensions, and exports:
  - results/model_comparison_baseline_vs_stgcn.csv
"""

import sys
import os
import argparse
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser(description="Compare Baseline vs ST-GCN.")
    parser.add_argument(
        "--baseline-json",
        type=str,
        default="results/baseline/metrics.json",
        help="Path to Baseline metrics.json"
    )
    parser.add_argument(
        "--stgcn-json",
        type=str,
        default="results/stgcn/metrics.json",
        help="Path to ST-GCN metrics.json"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/model_comparison_baseline_vs_stgcn.csv",
        help="Path to save comparison CSV"
    )
    args = parser.parse_args()

    if not os.path.exists(args.baseline_json) or not os.path.exists(args.stgcn_json):
        print(f"[ERROR] Both metrics JSON files must exist: {args.baseline_json} and {args.stgcn_json}")
        sys.exit(1)

    with open(args.baseline_json, "r", encoding="utf-8") as f:
        base_data = json.load(f)
    with open(args.stgcn_json, "r", encoding="utf-8") as f:
        stgcn_data = json.load(f)

    b_m = base_data.get("metrics", {})
    b_b = base_data.get("benchmark", {})
    s_m = stgcn_data.get("metrics", {})
    s_b = stgcn_data.get("benchmark", {})

    metrics_keys = [
        ("accuracy", "Top-1 Accuracy"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
        ("macro_f1", "Macro F1-Score"),
        ("weighted_f1", "Weighted F1-Score"),
    ]

    records = []

    # Model rows
    for model_name, m_dict, b_dict, sz in [
        ("Baseline (BiLSTM)", b_m, b_b, base_data.get("model_size_mb", 0.0)),
        ("ST-GCN (Graph Conv)", s_m, s_b, stgcn_data.get("model_size_mb", 0.0))
    ]:
        records.append({
            "model": model_name,
            "accuracy": m_dict.get("accuracy", None),
            "macro_precision": m_dict.get("macro_precision", None),
            "macro_recall": m_dict.get("macro_recall", None),
            "macro_f1": m_dict.get("macro_f1", None),
            "weighted_f1": m_dict.get("weighted_f1", None),
            "latency_ms": b_dict.get("mean_latency_ms", None),
            "fps": b_dict.get("throughput_fps", None),
            "parameters": b_dict.get("total_parameters", None),
            "model_size_mb": sz
        })

    # Absolute improvement row
    abs_imp = {
        "model": "Absolute Improvement (ST-GCN - Baseline)",
        "accuracy": round(s_m.get("accuracy", 0.0) - b_m.get("accuracy", 0.0), 4),
        "macro_precision": round(s_m.get("macro_precision", 0.0) - b_m.get("macro_precision", 0.0), 4),
        "macro_recall": round(s_m.get("macro_recall", 0.0) - b_m.get("macro_recall", 0.0), 4),
        "macro_f1": round(s_m.get("macro_f1", 0.0) - b_m.get("macro_f1", 0.0), 4),
        "weighted_f1": round(s_m.get("weighted_f1", 0.0) - b_m.get("weighted_f1", 0.0), 4),
        "latency_ms": round(s_b.get("mean_latency_ms", 0.0) - b_b.get("mean_latency_ms", 0.0), 2),
        "fps": round(s_b.get("throughput_fps", 0.0) - b_b.get("throughput_fps", 0.0), 2),
        "parameters": s_b.get("total_parameters", 0) - b_b.get("total_parameters", 0),
        "model_size_mb": round(stgcn_data.get("model_size_mb", 0.0) - base_data.get("model_size_mb", 0.0), 2)
    }
    records.append(abs_imp)

    # Relative improvement row
    def calc_rel(val_new, val_old):
        if val_old and val_old != 0:
            return round(((val_new - val_old) / abs(val_old)) * 100.0, 2)
        return None

    rel_imp = {
        "model": "Relative Improvement (%)",
        "accuracy": calc_rel(s_m.get("accuracy", 0.0), b_m.get("accuracy", 0.0)),
        "macro_precision": calc_rel(s_m.get("macro_precision", 0.0), b_m.get("macro_precision", 0.0)),
        "macro_recall": calc_rel(s_m.get("macro_recall", 0.0), b_m.get("macro_recall", 0.0)),
        "macro_f1": calc_rel(s_m.get("macro_f1", 0.0), b_m.get("macro_f1", 0.0)),
        "weighted_f1": calc_rel(s_m.get("weighted_f1", 0.0), b_m.get("weighted_f1", 0.0)),
        "latency_ms": calc_rel(s_b.get("mean_latency_ms", 0.0), b_b.get("mean_latency_ms", 0.0)),
        "fps": calc_rel(s_b.get("throughput_fps", 0.0), b_b.get("throughput_fps", 0.0)),
        "parameters": calc_rel(s_b.get("total_parameters", 0), b_b.get("total_parameters", 0)),
        "model_size_mb": calc_rel(stgcn_data.get("model_size_mb", 0.0), base_data.get("model_size_mb", 0.0))
    }
    records.append(rel_imp)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print("\n" + "=" * 80)
    print("           CONTROLLED MODEL COMPARISON: BASELINE vs ST-GCN")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    print(f"\n[SUCCESS] Comparison table exported to: {args.output_csv}\n")


if __name__ == "__main__":
    main()
