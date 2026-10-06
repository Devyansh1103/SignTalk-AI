"""
SignTalk AI: Unified Evaluation Metrics Dispatcher and Aggregator.

Provides a single interface to compute:
  - Classification metrics (Top-1, Top-3, Macro/Weighted Precision/Recall/F1, Per-Class stats)
  - Translation metrics (Token Acc, Sequence EM, BLEU-1 to 4, ROUGE-L, chrF, WER)
  - Confidence calibration metrics (ECE, Brier Score)
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import torch

from src.evaluation.classification_metrics import (
    compute_classification_metrics,
    format_metrics_table
)
from src.evaluation.translation_metrics import (
    compute_token_accuracy,
    compute_sequence_accuracy,
    compute_bleu,
    compute_rouge_l,
    compute_chrf,
    compute_wer
)


def compute_metrics(
    task_type: str,
    y_true: Union[np.ndarray, torch.Tensor, List[Any]],
    y_pred: Union[np.ndarray, torch.Tensor, List[Any]],
    y_probs: Optional[Union[np.ndarray, torch.Tensor]] = None,
    num_classes: int = 10,
    class_names: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Unified entry point for computing metrics according to task type.
    
    Args:
        task_type: 'classification' or 'translation' / 'sequence'
        y_true: Ground truth labels or token sequences
        y_pred: Predictions (class IDs or token sequences)
        y_probs: Optional class posterior probabilities
        num_classes: Number of classification targets
        class_names: Optional list of class names
        **kwargs: Additional parameters passed to specific metric calculators
        
    Returns:
        Dictionary of computed metrics.
    """
    if task_type == "classification":
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y_probs, torch.Tensor):
            y_probs = y_probs.detach().cpu().numpy()

        return compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_probs=y_probs,
            num_classes=num_classes,
            class_names=class_names
        )

    elif task_type in ("translation", "sequence"):
        pad_idx = kwargs.get("pad_idx", 0)
        eos_idx = kwargs.get("eos_idx", 3)

        if isinstance(y_true, list) and len(y_true) > 0 and isinstance(y_true[0], str):
            # Text strings
            hypotheses_words = [p.split() for p in y_pred]
            references_words = [t.split() for t in y_true]
            bleu = compute_bleu(hypotheses_words, references_words)
            rouge = compute_rouge_l(hypotheses_words, references_words)
            chrf = compute_chrf(y_pred, y_true)
            wer = compute_wer(hypotheses_words, references_words)
            em = float(np.mean([p.strip().lower() == t.strip().lower() for p, t in zip(y_pred, y_true)]))

            return {
                "sequence_accuracy": round(em, 4),
                "bleu": bleu,
                "rouge": rouge,
                "chrf": chrf,
                "wer": wer,
                "sample_count": len(y_true)
            }
        else:
            # Token ID tensors
            if not isinstance(y_true, torch.Tensor):
                y_true = torch.tensor(y_true)
            if not isinstance(y_pred, torch.Tensor):
                y_pred = torch.tensor(y_pred)

            token_acc = compute_token_accuracy(y_pred, y_true, pad_idx=pad_idx)
            seq_acc = compute_sequence_accuracy(y_pred, y_true, pad_idx=pad_idx, eos_idx=eos_idx)

            return {
                "token_accuracy": round(token_acc, 4),
                "sequence_accuracy": round(seq_acc, 4),
                "sample_count": int(y_true.shape[0])
            }
    else:
        raise ValueError(f"Unsupported task_type: '{task_type}'. Choose 'classification' or 'translation'.")
