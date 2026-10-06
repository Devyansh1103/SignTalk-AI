"""
SignTalk AI: Transformer Translation Layer Evaluation Script.

Evaluates the trained SignTranslationModel on the test partition:
  - Token-level metrics: Accuracy, Macro/Weighted Precision, Recall, F1
  - Sequence-level metrics: Exact Match (EM) Accuracy
  - Translation metrics: BLEU-1, BLEU-2, BLEU-4, ROUGE-L
  - Confidence calibration and per-sample audit
  - Generates evaluation_test.json and sample_predictions.csv
"""

import os
import sys
import argparse
import json
import time
from typing import Optional, List, Dict, Any

sys.path.insert(0, os.getcwd())

import yaml
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.sign_sequence_dataset import SignSequenceDataset
from src.data.collate import sign_sequence_collate_fn
from src.models.sign_translation_model import SignTranslationModel
from src.nlp.tokenizer import SignLanguageTokenizer
from src.evaluation.sequence_metrics import (
    compute_token_accuracy,
    compute_sequence_accuracy,
    compute_token_classification_metrics,
    compute_bleu,
    compute_rouge_l
)
from src.utils.device import resolve_device


def evaluate_transformer(
    config_path: str = "configs/transformer.yaml",
    checkpoint_path: Optional[str] = None
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(cfg.get("device", "auto"))
    paths = cfg["paths"]
    test_manifest = paths["test_manifest"]
    checkpoint_file = checkpoint_path or os.path.join(paths["checkpoint_dir"], "best_checkpoint.pt")
    metrics_dir = paths["metrics_dir"]
    os.makedirs(metrics_dir, exist_ok=True)

    print("=" * 70)
    print("SignTalk AI — Evaluating Sign Language Transformer on Test Set")
    print(f"Checkpoint: {checkpoint_file}")
    print(f"Test Manifest: {test_manifest}")
    print(f"Device: {device}")
    print("=" * 70)

    # 1. Load Tokenizer & Dataset
    tokenizer = SignLanguageTokenizer(vocab_path=cfg["transformer"]["vocabulary_file"])
    test_dataset = SignSequenceDataset(manifest_path=test_manifest, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Single sequence for exact latency and generation tracking
        shuffle=False,
        collate_fn=sign_sequence_collate_fn
    )

    print(f"Loaded {len(test_dataset)} test samples.")

    # 2. Instantiate Model and Load Checkpoint
    model = SignTranslationModel(
        stgcn_config=cfg.get("stgcn", {}),
        transformer_config=cfg.get("transformer", {}),
        freeze_stgcn=True
    ).to(device)

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    ckpt = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Model weights successfully restored.")

    pad_idx = cfg["transformer"].get("pad_idx", 0)
    bos_idx = cfg["transformer"].get("bos_idx", 2)
    eos_idx = cfg["transformer"].get("eos_idx", 3)

    all_teacher_preds = []
    all_targets = []
    all_gen_tokens = []
    all_gen_confs = []

    hypotheses_gloss = []
    references_gloss = []
    hypotheses_trans = []
    references_trans = []

    sample_audit = []
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    total_test_loss = 0.0

    t_start = time.perf_counter()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            dec_in = batch["decoder_input"].to(device)
            dec_tgt = batch["decoder_target"].to(device)
            meta = batch["metadata"][0]

            t_sample_start = time.perf_counter()

            # Teacher-forced evaluation
            logits = model(x=x, target_tokens=dec_in, mask=mask)
            B, S, V = logits.shape
            loss = criterion(logits.view(-1, V), dec_tgt.view(-1))
            total_test_loss += loss.item()

            teacher_pred = logits.argmax(dim=-1)
            all_teacher_preds.append(teacher_pred.cpu())
            all_targets.append(dec_tgt.cpu())

            # Autoregressive generation
            gen_tokens, confidences = model.generate(x=x, mask=mask, max_length=5)
            sample_ms = (time.perf_counter() - t_sample_start) * 1000.0

            gen_tok_list = gen_tokens[0].cpu().tolist()
            conf_list = confidences[0].cpu().tolist()

            # Extract predicted lexical token (excluding BOS, EOS, PAD)
            lex_tokens = [t for t in gen_tok_list if t not in {pad_idx, bos_idx, eos_idx}]
            lex_confs = [c for t, c in zip(gen_tok_list, conf_list) if t not in {pad_idx, bos_idx, eos_idx}]

            pred_tok_id = lex_tokens[0] if lex_tokens else tokenizer.unk_id
            pred_conf = float(lex_confs[0]) if lex_confs else 0.0

            gt_tok_id = dec_tgt[0, 0].item()  # First target token before EOS

            pred_gloss = tokenizer.decode([pred_tok_id], skip_special_tokens=True)
            pred_trans = tokenizer.decode_to_translation([pred_tok_id])

            gt_gloss = str(meta["gloss"])
            gt_trans = str(meta["translation"])

            is_correct = (pred_tok_id == gt_tok_id)

            hypotheses_gloss.append([pred_gloss])
            references_gloss.append([gt_gloss])
            hypotheses_trans.append(pred_trans.split())
            references_trans.append(gt_trans.split())

            sample_audit.append({
                "sample_id": meta["sequence_id"],
                "signer_id": meta["signer_id"],
                "ground_truth_gloss": gt_gloss,
                "predicted_gloss": pred_gloss,
                "ground_truth_translation": gt_trans,
                "predicted_translation": pred_trans,
                "gt_token_id": gt_tok_id,
                "pred_token_id": pred_tok_id,
                "confidence": round(pred_conf, 4),
                "is_correct": is_correct,
                "latency_ms": round(sample_ms, 2)
            })

            all_gen_tokens.append(gen_tokens[:, 1:].cpu())  # drop BOS for sequence match

    total_eval_time = time.perf_counter() - t_start

    # Compute Aggregate Metrics
    cat_preds = torch.cat(all_teacher_preds, dim=0)
    cat_targets = torch.cat(all_targets, dim=0)
    cat_gen = torch.cat(all_gen_tokens, dim=0)

    token_acc = compute_token_accuracy(cat_preds, cat_targets, pad_idx=pad_idx)
    teacher_seq_acc = compute_sequence_accuracy(cat_preds, cat_targets, pad_idx=pad_idx)
    gen_seq_acc = compute_sequence_accuracy(cat_gen, cat_targets, pad_idx=pad_idx)
    class_metrics = compute_token_classification_metrics(cat_preds, cat_targets, pad_idx=pad_idx)

    # Translation Metrics
    bleu_metrics = compute_bleu(hypotheses_trans, references_trans, max_order=4)
    rouge_metrics = compute_rouge_l(hypotheses_trans, references_trans)

    mean_test_loss = total_test_loss / len(test_loader)

    results = {
        "evaluation_partition": "test",
        "sample_count": len(test_dataset),
        "test_loss": round(mean_test_loss, 4),
        "token_accuracy": round(token_acc, 4),
        "teacher_forced_sequence_accuracy": round(teacher_seq_acc, 4),
        "generated_sequence_accuracy": round(gen_seq_acc, 4),
        "macro_precision": round(class_metrics["macro_precision"], 4),
        "macro_recall": round(class_metrics["macro_recall"], 4),
        "macro_f1": round(class_metrics["macro_f1"], 4),
        "weighted_f1": round(class_metrics["weighted_f1"], 4),
        "bleu": bleu_metrics,
        "rouge": rouge_metrics,
        "mean_latency_ms": round(float(pd.DataFrame(sample_audit)["latency_ms"].mean()), 2),
        "median_latency_ms": round(float(pd.DataFrame(sample_audit)["latency_ms"].median()), 2),
        "p95_latency_ms": round(float(pd.DataFrame(sample_audit)["latency_ms"].quantile(0.95)), 2)
    }

    # Print summary
    print("\n" + "=" * 70)
    print("TEST EVALUATION RESULTS:")
    print("=" * 70)
    print(f"Generated Sequence Exact Match: {gen_seq_acc*100:.2f}% ({int(gen_seq_acc*len(test_dataset))}/{len(test_dataset)})")
    print(f"Token Accuracy:                 {token_acc*100:.2f}%")
    print(f"Macro Precision:                {class_metrics['macro_precision']*100:.2f}%")
    print(f"Macro Recall:                   {class_metrics['macro_recall']*100:.2f}%")
    print(f"Macro F1-Score:                 {class_metrics['macro_f1']*100:.2f}%")
    print(f"Weighted F1-Score:              {class_metrics['weighted_f1']*100:.2f}%")
    print(f"BLEU-1 Score:                   {bleu_metrics['bleu_1']*100:.2f}%")
    print(f"ROUGE-L F1:                     {rouge_metrics['rouge_l_f1']*100:.2f}%")
    print(f"Mean Latency per Sample:        {results['mean_latency_ms']:.2f} ms")
    print("=" * 70)

    # Save JSON and CSV
    json_path = os.path.join(metrics_dir, "evaluation_test.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(metrics_dir, "sample_predictions.csv")
    pd.DataFrame(sample_audit).to_csv(csv_path, index=False)

    print(f"Saved evaluation metrics to: {json_path}")
    print(f"Saved sample audit to:       {csv_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Sign Language Transformer")
    parser.add_argument("--config", type=str, default="configs/transformer.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    evaluate_transformer(args.config, args.checkpoint)
