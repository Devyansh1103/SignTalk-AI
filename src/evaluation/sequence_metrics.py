"""
SignTalk AI: Sequence & Translation Evaluation Metrics.

Computes token-level, sequence-level, and translation-level metrics:
  - Token Accuracy (ignoring padding)
  - Sequence Exact Match Accuracy
  - Macro / Weighted Precision, Recall, and F1
  - BLEU-1, BLEU-2, BLEU-4
  - ROUGE-L (LCS-based recall, precision, and F-measure)
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import math
from collections import Counter
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support


def compute_token_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int = 0
) -> float:
    """
    Computes token-level accuracy over all non-padding positions.
    
    Args:
        predictions: Tensor of shape [B, S] containing predicted token IDs.
        targets: Tensor of shape [B, S] containing target token IDs.
        pad_idx: Token ID to ignore in evaluation.
        
    Returns:
        Float accuracy in [0.0, 1.0].
    """
    valid_mask = targets != pad_idx
    if valid_mask.sum() == 0:
        return 0.0
    correct = (predictions == targets) & valid_mask
    return float(correct.sum().item() / valid_mask.sum().item())


def compute_sequence_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int = 0,
    eos_idx: int = 3
) -> float:
    """
    Computes sequence exact match (EM) accuracy:
    A sequence is correct if and only if all tokens up to <EOS> match exactly.
    
    Args:
        predictions: Tensor of shape [B, S] (or list of sequences).
        targets: Tensor of shape [B, S] (or list of sequences).
        pad_idx: Padding token ID.
        eos_idx: End-of-sequence token ID.
        
    Returns:
        Float exact-match accuracy in [0.0, 1.0].
    """
    if isinstance(predictions, torch.Tensor):
        preds_list = predictions.detach().cpu().tolist()
    else:
        preds_list = predictions

    if isinstance(targets, torch.Tensor):
        tgts_list = targets.detach().cpu().tolist()
    else:
        tgts_list = targets

    correct_count = 0
    total = len(preds_list)
    if total == 0:
        return 0.0

    for p_seq, t_seq in zip(preds_list, tgts_list):
        # Truncate at EOS or PAD
        p_clean = []
        for tok in p_seq:
            if tok == eos_idx:
                p_clean.append(tok)
                break
            if tok != pad_idx:
                p_clean.append(tok)

        t_clean = []
        for tok in t_seq:
            if tok == eos_idx:
                t_clean.append(tok)
                break
            if tok != pad_idx:
                t_clean.append(tok)

        if p_clean == t_clean:
            correct_count += 1

    return float(correct_count / total)


def compute_token_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int = 0
) -> Dict[str, float]:
    """
    Computes macro and weighted precision, recall, and F1 across non-padding tokens.
    """
    preds_flat = predictions.detach().cpu().numpy().flatten()
    tgts_flat = targets.detach().cpu().numpy().flatten()

    valid_mask = tgts_flat != pad_idx
    if not np.any(valid_mask):
        return {
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0
        }

    y_true = tgts_flat[valid_mask]
    y_pred = preds_flat[valid_mask]

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1)
    }


def compute_ngram_clip(
    candidate: List[str],
    reference: List[str],
    n: int
) -> Tuple[int, int]:
    """Computes clipped n-gram counts for BLEU."""
    if len(candidate) < n or len(reference) < n:
        return 0, max(0, len(candidate) - n + 1)

    cand_ngrams = Counter(tuple(candidate[i:i+n]) for i in range(len(candidate) - n + 1))
    ref_ngrams = Counter(tuple(reference[i:i+n]) for i in range(len(reference) - n + 1))

    clipped_count = sum(min(count, ref_ngrams[ng]) for ng, count in cand_ngrams.items())
    total_cand = sum(cand_ngrams.values())
    return clipped_count, total_cand


def compute_bleu(
    hypotheses: List[List[str]],
    references: List[List[str]],
    max_order: int = 4
) -> Dict[str, float]:
    """
    Computes standard sentence/corpus BLEU scores (BLEU-1, BLEU-2, BLEU-4).
    """
    if len(hypotheses) != len(references) or len(hypotheses) == 0:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_4": 0.0}

    total_clipped = [0] * max_order
    total_cand = [0] * max_order
    cand_len = 0
    ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        cand_len += len(hyp)
        ref_len += len(ref)

        for n in range(1, max_order + 1):
            clip, tot = compute_ngram_clip(hyp, ref, n)
            total_clipped[n - 1] += clip
            total_cand[n - 1] += tot

    precisions = []
    for n in range(max_order):
        if total_cand[n] > 0:
            precisions.append(total_clipped[n] / total_cand[n])
        else:
            precisions.append(0.0)

    # Brevity penalty
    if cand_len == 0:
        bp = 0.0
    elif cand_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0

    bleu_1 = bp * precisions[0] if precisions[0] > 0 else 0.0
    
    # BLEU-2 geometric mean
    if precisions[0] > 0 and precisions[1] > 0:
        bleu_2 = bp * math.exp(0.5 * (math.log(precisions[0]) + math.log(precisions[1])))
    else:
        bleu_2 = 0.0

    # BLEU-4 geometric mean
    if all(p > 0 for p in precisions[:4]):
        log_mean = sum(0.25 * math.log(p) for p in precisions[:4])
        bleu_4 = bp * math.exp(log_mean)
    else:
        bleu_4 = 0.0

    return {
        "bleu_1": float(round(bleu_1, 4)),
        "bleu_2": float(round(bleu_2, 4)),
        "bleu_4": float(round(bleu_4, 4))
    }


def compute_lcs(seq1: List[str], seq2: List[str]) -> int:
    """Computes longest common subsequence length."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge_l(
    hypotheses: List[List[str]],
    references: List[List[str]]
) -> Dict[str, float]:
    """Computes ROUGE-L (recall, precision, F1)."""
    if len(hypotheses) == 0:
        return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0}

    precisions = []
    recalls = []
    f1s = []

    for hyp, ref in zip(hypotheses, references):
        lcs = compute_lcs(hyp, ref)
        p = lcs / len(hyp) if len(hyp) > 0 else 0.0
        r = lcs / len(ref) if len(ref) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    return {
        "rouge_l_precision": float(round(np.mean(precisions), 4)),
        "rouge_l_recall": float(round(np.mean(recalls), 4)),
        "rouge_l_f1": float(round(np.mean(f1s), 4))
    }
