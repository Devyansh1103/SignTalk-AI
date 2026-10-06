"""
SignTalk AI: Unified Model Evaluator Engine.

Standardizes evaluation across:
  1. Baseline (SignBaselineModel)
  2. ST-GCN (SignSTGCN)
  3. ST-GCN + Transformer (SignTranslationModel)

Produces standardized artifacts:
  - metrics.json
  - predictions.csv
  - confusion_matrix.png
  - benchmark.json
  - error_report.json
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.device import get_device, resolve_device
from src.models.baseline import SignBaselineModel
from src.models.stgcn import SignSTGCN
from src.models.sign_translation_model import SignTranslationModel
from src.data.dataloader import create_sequence_dataloader
from src.evaluation.classification_metrics import compute_classification_metrics, format_metrics_table
from src.evaluation.translation_metrics import (
    compute_token_accuracy,
    compute_sequence_accuracy,
    compute_bleu,
    compute_rouge_l,
    compute_chrf,
    compute_wer
)
from src.evaluation.confusion_matrix import (
    compute_confusion_matrix,
    analyze_top_confusions,
    plot_confusion_matrix
)
from src.evaluation.confidence_analysis import (
    compute_expected_calibration_error,
    compute_brier_score,
    analyze_threshold_rejection,
    summarize_confidence_distribution
)
from src.evaluation.benchmark import (
    benchmark_model_inference,
    get_hardware_environment,
    count_parameters
)
from src.evaluation.error_analysis import ErrorAnalyzer


class ModelEvaluator:
    """Standardized evaluation controller for all SignTalk AI candidate models."""

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "stgcn",
        device: str = "auto",
        class_names: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type.lower()
        self.device, self.device_info = get_device(device)
        self.class_names = class_names or [
            "hello", "thankyou", "good", "happy", "monday",
            "car", "bird", "house", "time", "teacher"
        ]
        self.config = config or {}

        # Load checkpoint and instantiate model
        self.ckpt = None
        self.model = self._load_model()
        self.hardware_env = get_hardware_environment()
        self.error_analyzer = ErrorAnalyzer(class_names=self.class_names)

    def _load_model(self) -> nn.Module:
        """Loads and reconstructs model from checkpoint."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        self.ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        ckpt_config = self.ckpt.get("config", {})
        model_cfg = ckpt_config.get("model", {})

        if self.model_type == "baseline":
            model = SignBaselineModel(
                in_channels=model_cfg.get("in_channels", 3),
                num_nodes=model_cfg.get("num_nodes", 93),
                sequence_length=model_cfg.get("sequence_length", 45),
                proj_dim=model_cfg.get("proj_dim", 128),
                hidden_dim=model_cfg.get("hidden_dim", 128),
                num_layers=model_cfg.get("num_layers", 2),
                dropout=model_cfg.get("dropout", 0.3),
                bidirectional=model_cfg.get("bidirectional", True),
                num_classes=model_cfg.get("num_classes", len(self.class_names)),
                pooling_type=model_cfg.get("pooling_type", "mean_max")
            )
            model.load_state_dict(self.ckpt["model_state_dict"])

        elif self.model_type == "stgcn":
            model = SignSTGCN(
                in_channels=model_cfg.get("in_channels", 3),
                num_classes=model_cfg.get("num_classes", len(self.class_names)),
                num_nodes=model_cfg.get("num_nodes", 93),
                sequence_length=model_cfg.get("sequence_length", 45),
                graph_strategy=model_cfg.get("graph_strategy", "spatial"),
                block_channels=model_cfg.get("block_channels", [64, 64, 128, 128, 256, 256]),
                block_strides=model_cfg.get("block_strides", [1, 1, 2, 1, 2, 1]),
                temporal_kernel_size=model_cfg.get("temporal_kernel_size", 9),
                dropout=model_cfg.get("dropout", 0.3),
                residual=model_cfg.get("residual", True),
                use_learnable_edge_weights=model_cfg.get("use_learnable_edge_weights", True)
            )
            model.load_state_dict(self.ckpt["model_state_dict"])

        elif self.model_type in ("transformer", "translation"):
            stgcn_cfg = ckpt_config.get("stgcn", {})
            trans_cfg = ckpt_config.get("transformer", {})
            model = SignTranslationModel(
                stgcn_config=stgcn_cfg,
                transformer_config=trans_cfg,
                freeze_stgcn=ckpt_config.get("training", {}).get("freeze_stgcn", True)
            )
            model.load_state_dict(self.ckpt["model_state_dict"])

        else:
            raise ValueError(f"Unknown model_type: '{self.model_type}'. Choose 'baseline', 'stgcn', or 'transformer'.")

        model.to(self.device)
        model.eval()
        return model

    def evaluate(
        self,
        dataloader: DataLoader,
        split_name: str = "test",
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes complete evaluation suite on provided dataloader.
        """
        if self.model_type in ("baseline", "stgcn"):
            return self._evaluate_classification(dataloader, split_name, output_dir)
        else:
            return self._evaluate_translation(dataloader, split_name, output_dir)

    def _evaluate_classification(
        self,
        dataloader: DataLoader,
        split_name: str,
        output_dir: Optional[str]
    ) -> Dict[str, Any]:
        """Runs classification evaluation pipeline."""
        all_targets = []
        all_preds = []
        all_probs = []
        metadata_list = []
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in dataloader:
                x = (batch.get("x") if "x" in batch else batch.get("landmarks")).to(self.device)
                y = batch["label"].to(self.device)
                mask = batch.get("mask", None)
                if mask is not None:
                    mask = mask.to(self.device)

                logits = self.model(x, mask)
                loss = criterion(logits, y)
                total_loss += loss.item() * x.size(0)

                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                all_targets.extend(y.cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

                # Collect sample metadata
                batch_size = x.size(0)
                batch_meta = batch.get("metadata", None)
                for b in range(batch_size):
                    if batch_meta and b < len(batch_meta):
                        m_dict = dict(batch_meta[b])
                        meta = {
                            "sequence_id": m_dict.get("sequence_id", f"seq_{len(metadata_list):04d}"),
                            "signer_id": m_dict.get("signer_id", "unknown"),
                            "valid_frames": int(m_dict.get("valid_frames", 45)),
                            "quality_score": float(m_dict.get("quality_score", 0.7)),
                            "quality_status": str(m_dict.get("quality_status", "ACCEPTABLE")),
                            "pose_detection_rate": float(m_dict.get("pose_detection_rate", 100.0)),
                            "left_hand_detection_rate": float(m_dict.get("left_hand_detection_rate", 50.0)),
                            "right_hand_detection_rate": float(m_dict.get("right_hand_detection_rate", 50.0)),
                            "missing_landmark_ratio": float(m_dict.get("missing_landmark_ratio", 0.5))
                        }
                    else:
                        meta = {
                            "sequence_id": f"seq_{len(metadata_list):04d}",
                            "signer_id": "unknown",
                            "valid_frames": 45,
                            "quality_score": 0.7,
                            "quality_status": "ACCEPTABLE",
                            "pose_detection_rate": 100.0,
                            "left_hand_detection_rate": 50.0,
                            "right_hand_detection_rate": 50.0,
                            "missing_landmark_ratio": 0.5
                        }
                    metadata_list.append(meta)

        N = len(all_targets)
        avg_loss = total_loss / max(N, 1)
        y_true = np.array(all_targets, dtype=np.int64)
        y_pred = np.array(all_preds, dtype=np.int64)
        y_probs = np.array(all_probs, dtype=np.float64)

        # 1. Classification Metrics
        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_probs=y_probs,
            num_classes=len(self.class_names),
            class_names=self.class_names
        )
        metrics["loss"] = round(avg_loss, 4)

        # 2. Confusion Analysis
        cm_raw, cm_norm = compute_confusion_matrix(y_true, y_pred, num_classes=len(self.class_names))
        top_confusions = analyze_top_confusions(cm_raw, self.class_names, top_k=5)

        # 3. Confidence & Calibration
        confidences = np.max(y_probs, axis=1)
        accuracies = (y_true == y_pred)
        ece, ece_bins = compute_expected_calibration_error(confidences, accuracies)
        brier = compute_brier_score(y_true, y_probs, num_classes=len(self.class_names))
        conf_dist = summarize_confidence_distribution(confidences, accuracies)
        rejection_analysis = analyze_threshold_rejection(confidences, accuracies)

        # 4. Latency Benchmark
        dummy_x = torch.zeros(1, 3, 45, 93)
        bench = benchmark_model_inference(self.model, dummy_x, self.device, num_warmup=10, num_iterations=50)

        # 5. Error Categorization
        errors_df = self.error_analyzer.analyze_dataset_errors(
            y_true=all_targets,
            y_pred=all_preds,
            confidences=confidences.tolist(),
            metadata_list=metadata_list
        )
        error_summary = self.error_analyzer.summarize_categories(errors_df)

        # 6. Checkpoint size
        ckpt_size_mb = os.path.getsize(self.checkpoint_path) / (1024 * 1024) if os.path.exists(self.checkpoint_path) else 0.0

        full_results = {
            "model_type": self.model_type,
            "checkpoint": self.checkpoint_path,
            "split": split_name,
            "sample_count": N,
            "metrics": metrics,
            "calibration": {
                "ece": ece,
                "brier_score": brier,
                "confidence_distribution": conf_dist,
                "threshold_rejection": rejection_analysis
            },
            "top_confusions": top_confusions,
            "error_analysis": error_summary,
            "benchmark": bench,
            "model_size_mb": round(ckpt_size_mb, 2),
            "hardware": self.hardware_env
        }

        # 7. Predictions DataFrame
        pred_records = []
        for i in range(N):
            m = metadata_list[i] if i < len(metadata_list) else {}
            pred_records.append({
                "sample_idx": i,
                "sequence_id": m.get("sequence_id", f"seq_{i:04d}"),
                "signer_id": m.get("signer_id", "unknown"),
                "true_class_id": int(y_true[i]),
                "true_label": self.class_names[y_true[i]],
                "pred_class_id": int(y_pred[i]),
                "pred_label": self.class_names[y_pred[i]],
                "is_correct": bool(y_true[i] == y_pred[i]),
                "confidence": round(float(confidences[i]), 4),
                "quality_status": m.get("quality_status", "ACCEPTABLE"),
                "valid_frames": m.get("valid_frames", 45)
            })
        pred_df = pd.DataFrame(pred_records)

        # Save artifacts if output_dir provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(full_results, f, indent=2)

            pred_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

            cm_path = os.path.join(output_dir, "confusion_matrix.png")
            plot_confusion_matrix(
                cm_norm=cm_norm,
                class_names=self.class_names,
                output_path=cm_path,
                title=f"{self.model_type.upper()} Normalized Confusion Matrix ({split_name.upper()})"
            )

            with open(os.path.join(output_dir, "benchmark.json"), "w", encoding="utf-8") as f:
                json.dump(bench, f, indent=2)

            errors_df.to_csv(os.path.join(output_dir, "errors.csv"), index=False)

        return full_results

    def _evaluate_translation(
        self,
        dataloader: DataLoader,
        split_name: str,
        output_dir: Optional[str]
    ) -> Dict[str, Any]:
        """Runs autoregressive translation evaluation pipeline."""
        from src.nlp.tokenizer import SignLanguageTokenizer
        tokenizer = SignLanguageTokenizer()

        all_preds = []
        all_targets = []
        all_hyp_texts = []
        all_ref_texts = []
        latencies = []

        with torch.no_grad():
            for batch in dataloader:
                x = (batch.get("x") if "x" in batch else batch.get("landmarks")).to(self.device)
                y_tokens = batch.get("tokens", None)
                batch_meta = batch.get("metadata", None)

                if y_tokens is None:
                    if "decoder_target" in batch:
                        y_tokens = batch["decoder_target"].to(self.device)
                    else:
                        glosses = batch.get("gloss", [])
                        token_list = [tokenizer.encode(g, add_special_tokens=True) for g in glosses]
                        y_tokens = torch.nn.utils.rnn.pad_sequence(
                            [torch.tensor(t) for t in token_list],
                            batch_first=True,
                            padding_value=tokenizer.pad_token_id
                        ).to(self.device)
                else:
                    y_tokens = y_tokens.to(self.device)

                mask = batch.get("mask", None)
                if mask is not None:
                    mask = mask.to(self.device)

                # Time autoregressive generation
                t0 = time.perf_counter()
                gen_res = self.model.generate(
                    x,
                    mask=mask,
                    max_length=6
                )
                if isinstance(gen_res, tuple):
                    gen_tokens = gen_res[0]
                else:
                    gen_tokens = gen_res
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0 / max(x.size(0), 1))

                # Drop BOS from generated tokens to align with decoder_target [gloss_id, EOS]
                gen_aligned = gen_tokens[:, 1:] if gen_tokens.size(1) > 1 else gen_tokens
                all_preds.append(gen_aligned.cpu())
                all_targets.append(y_tokens.cpu())

                for b in range(x.size(0)):
                    hyp_str = tokenizer.decode(gen_tokens[b].tolist(), skip_special_tokens=True)
                    ref_str = tokenizer.decode(y_tokens[b].tolist(), skip_special_tokens=True)
                    all_hyp_texts.append(hyp_str)
                    all_ref_texts.append(ref_str)

        preds_cat = torch.cat(all_preds, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)

        # Align lengths if needed
        max_len = max(preds_cat.size(1), targets_cat.size(1))
        if preds_cat.size(1) < max_len:
            pad = torch.full((preds_cat.size(0), max_len - preds_cat.size(1)), tokenizer.pad_id, dtype=torch.long)
            preds_cat = torch.cat([preds_cat, pad], dim=1)
        if targets_cat.size(1) < max_len:
            pad = torch.full((targets_cat.size(0), max_len - targets_cat.size(1)), tokenizer.pad_id, dtype=torch.long)
            targets_cat = torch.cat([targets_cat, pad], dim=1)

        token_acc = compute_token_accuracy(preds_cat, targets_cat, pad_idx=tokenizer.pad_id)
        seq_acc = compute_sequence_accuracy(preds_cat, targets_cat, pad_idx=tokenizer.pad_id, eos_idx=tokenizer.eos_id)

        hyp_words = [t.split() for t in all_hyp_texts]
        ref_words = [t.split() for t in all_ref_texts]
        bleu = compute_bleu(hyp_words, ref_words)
        rouge = compute_rouge_l(hyp_words, ref_words)
        chrf = compute_chrf(all_hyp_texts, all_ref_texts)
        wer = compute_wer(hyp_words, ref_words)

        mean_lat = float(np.mean(latencies))
        median_lat = float(np.median(latencies))
        p95_lat = float(np.percentile(latencies, 95))

        total_params, trainable_params = count_parameters(self.model)
        ckpt_size_mb = os.path.getsize(self.checkpoint_path) / (1024 * 1024) if os.path.exists(self.checkpoint_path) else 0.0

        full_results = {
            "model_type": self.model_type,
            "checkpoint": self.checkpoint_path,
            "split": split_name,
            "sample_count": len(all_hyp_texts),
            "metrics": {
                "token_accuracy": round(token_acc, 4),
                "sequence_exact_match": round(seq_acc, 4),
                "bleu": bleu,
                "rouge": rouge,
                "chrf": round(chrf, 4),
                "wer": round(wer, 4)
            },
            "benchmark": {
                "mean_latency_ms": round(mean_lat, 2),
                "median_latency_ms": round(median_lat, 2),
                "p95_latency_ms": round(p95_lat, 2),
                "throughput_fps": round(1000.0 / mean_lat, 2) if mean_lat > 0 else 0.0,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params
            },
            "model_size_mb": round(ckpt_size_mb, 2),
            "hardware": self.hardware_env
        }

        # Qualitative dataframe
        qual_records = []
        for i in range(len(all_hyp_texts)):
            qual_records.append({
                "sample_idx": i,
                "reference_gloss": all_ref_texts[i],
                "generated_gloss": all_hyp_texts[i],
                "exact_match": bool(all_ref_texts[i].strip() == all_hyp_texts[i].strip()),
                "error_type": "NONE" if (all_ref_texts[i].strip() == all_hyp_texts[i].strip()) else "TOKEN_SUBSTITUTION"
            })
        qual_df = pd.DataFrame(qual_records)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(full_results, f, indent=2)

            qual_df.to_csv(os.path.join(output_dir, "qualitative_examples.csv"), index=False)

            with open(os.path.join(output_dir, "benchmark.json"), "w", encoding="utf-8") as f:
                json.dump(full_results["benchmark"], f, indent=2)

        return full_results
