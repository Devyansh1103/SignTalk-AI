"""
SignTalk AI: Comprehensive Translation and Sequence Evaluation Metrics.

Computes:
  - Token Accuracy (excluding padding)
  - Sequence Exact Match (EM) Accuracy
  - Macro and Weighted Precision, Recall, and F1 over tokens
  - BLEU-1, BLEU-2, BLEU-3, BLEU-4
  - ROUGE-1, ROUGE-2, ROUGE-L
  - chrF (character n-gram F-score)
  - Word Error Rate (WER) and Character Error Rate (CER)
  - Qualitative translation analysis generator
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
    """Computes token accuracy over non-padding positions."""
    if isinstance(predictions, list):
        predictions = torch.tensor(predictions)
    if isinstance(targets, list):
        targets = torch.tensor(targets)

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
    """Computes sequence exact match (EM) accuracy up to EOS."""
    if isinstance(predictions, list):
        predictions = torch.tensor(predictions)
    if isinstance(targets, list):
        targets = torch.tensor(targets)

    batch_size = targets.shape[0]
    if batch_size == 0:
        return 0.0

    exact_matches = 0
    for b in range(batch_size):
        pred_seq = predictions[b].tolist()
        tgt_seq = targets[b].tolist()

        # Truncate at EOS or first PAD
        pred_clean = []
        for tok in pred_seq:
            if tok in (eos_idx, pad_idx):
                break
            pred_clean.append(tok)

        tgt_clean = []
        for tok in tgt_seq:
            if tok in (eos_idx, pad_idx):
                break
            tgt_clean.append(tok)

        if pred_clean == tgt_clean:
            exact_matches += 1

    return float(exact_matches / batch_size)


def compute_bleu(
    hypotheses: List[List[str]],
    references: List[List[str]],
    max_n: int = 4,
    smoothing: bool = True
) -> Dict[str, float]:
    """Computes sentence-level smoothed BLEU scores (BLEU-1 through BLEU-4)."""
    scores = {f"bleu_{n}": 0.0 for n in range(1, max_n + 1)}
    N = len(hypotheses)
    if N == 0 or len(references) == 0:
        return scores

    total_bleu_n = {n: 0.0 for n in range(1, max_n + 1)}

    for hyp, ref in zip(hypotheses, references):
        hyp_len = len(hyp)
        ref_len = len(ref)
        if hyp_len == 0:
            continue

        # Brevity penalty
        bp = 1.0 if hyp_len >= ref_len else math.exp(1.0 - ref_len / max(hyp_len, 1))

        # Precision for each n-gram
        precisions = []
        for n in range(1, max_n + 1):
            if hyp_len < n:
                prec = 0.0
            else:
                hyp_ngrams = [tuple(hyp[i:i+n]) for i in range(hyp_len - n + 1)]
                ref_ngrams = [tuple(ref[i:i+n]) for i in range(ref_len - n + 1)]
                hyp_counts = Counter(hyp_ngrams)
                ref_counts = Counter(ref_ngrams)

                clipped = sum(min(count, ref_counts.get(ng, 0)) for ng, count in hyp_counts.items())
                total = len(hyp_ngrams)
                if smoothing:
                    prec = (clipped + 0.1) / (total + 0.1)
                else:
                    prec = clipped / total if total > 0 else 0.0
            precisions.append(prec)

        # Compute BLEU-n
        for n in range(1, max_n + 1):
            subset = precisions[:n]
            if all(p > 0 for p in subset):
                log_sum = sum(math.log(p) for p in subset) / n
                bleu_score = bp * math.exp(log_sum)
            else:
                bleu_score = 0.0
            total_bleu_n[n] += bleu_score

    for n in range(1, max_n + 1):
        scores[f"bleu_{n}"] = round(total_bleu_n[n] / N, 4)

    return scores


def _lcs_length(s1: List[str], s2: List[str]) -> int:
    """Computes Length of Longest Common Subsequence."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge_l(
    hypotheses: List[List[str]],
    references: List[List[str]],
    beta: float = 1.2
) -> Dict[str, float]:
    """Computes ROUGE-L precision, recall, and F1."""
    N = len(hypotheses)
    if N == 0:
        return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0}

    total_p, total_r, total_f = 0.0, 0.0, 0.0
    for hyp, ref in zip(hypotheses, references):
        lcs = _lcs_length(hyp, ref)
        m, n = len(hyp), len(ref)

        prec = lcs / m if m > 0 else 0.0
        rec = lcs / n if n > 0 else 0.0

        if prec + rec > 0:
            f = ((1 + beta**2) * prec * rec) / ((beta**2 * prec) + rec)
        else:
            f = 0.0

        total_p += prec
        total_r += rec
        total_f += f

    return {
        "rouge_l_precision": round(total_p / N, 4),
        "rouge_l_recall": round(total_r / N, 4),
        "rouge_l_f1": round(total_f / N, 4)
    }


def compute_chrf(
    hypotheses: List[str],
    references: List[str],
    char_order: int = 6,
    beta: float = 2.0
) -> float:
    """Computes character n-gram F-score (chrF)."""
    if len(hypotheses) == 0:
        return 0.0

    total_f = 0.0
    for hyp_str, ref_str in zip(hypotheses, references):
        hyp_str = hyp_str.replace(" ", "")
        ref_str = ref_str.replace(" ", "")

        if len(hyp_str) == 0 and len(ref_str) == 0:
            total_f += 1.0
            continue
        if len(hyp_str) == 0 or len(ref_str) == 0:
            continue

        n_scores = []
        for n in range(1, char_order + 1):
            hyp_ngrams = [hyp_str[i:i+n] for i in range(len(hyp_str) - n + 1)]
            ref_ngrams = [ref_str[i:i+n] for i in range(len(ref_str) - n + 1)]

            hyp_counts = Counter(hyp_ngrams)
            ref_counts = Counter(ref_ngrams)

            clipped = sum(min(c, ref_counts.get(ng, 0)) for ng, c in hyp_counts.items())
            p = clipped / len(hyp_ngrams) if len(hyp_ngrams) > 0 else 0.0
            r = clipped / len(ref_ngrams) if len(ref_ngrams) > 0 else 0.0

            if p + r > 0:
                f_n = ((1 + beta**2) * p * r) / ((beta**2 * p) + r)
            else:
                f_n = 0.0
            n_scores.append(f_n)

        total_f += sum(n_scores) / len(n_scores) if n_scores else 0.0

    return round(total_f / len(hypotheses), 4)


def compute_wer(hypotheses: List[List[str]], references: List[List[str]]) -> float:
    """Computes Word Error Rate using Levenshtein distance on words."""
    total_dist = 0
    total_ref_words = 0

    for hyp, ref in zip(hypotheses, references):
        total_ref_words += len(ref)
        m, n = len(hyp), len(ref)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if hyp[i - 1] == ref[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        total_dist += dp[m][n]

    return round(total_dist / max(total_ref_words, 1), 4)
