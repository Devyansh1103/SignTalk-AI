"""
SignTalk AI: Sequence Batch Collation Module.

Collates individual sequence dictionary samples into batched PyTorch tensors:
  - Standard static batching: [B, C, T, V] = [B, 3, 45, 93]
  - Dynamic padding collate for variable-length sequences (if applicable)
  - Transformer target batching: [B, S]
"""

from typing import List, Dict, Any
import torch


def sign_sequence_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Standard batch collation function for fixed-length sequence objects.
    
    Returns:
      {
        "x": Tensor [B, C, 45, 93],
        "mask": Tensor [B, 1, 45, 93],
        "label": Tensor [B],
        "gloss_id": Tensor [B],
        "decoder_input": Tensor [B, 2],
        "decoder_target": Tensor [B, 2],
        "attention_mask": Tensor [B, 2],
        "metadata": List[Dict[str, Any]]
      }
    """
    x_list = [item["x"] for item in batch]
    mask_list = [item["mask"] for item in batch]
    label_list = [item["label"] for item in batch]
    gloss_list = [item["gloss_id"] for item in batch]
    dec_in_list = [item["decoder_input"] for item in batch]
    dec_tgt_list = [item["decoder_target"] for item in batch]
    attn_list = [item["attention_mask"] for item in batch]
    meta_list = [item["metadata"] for item in batch]

    batched_x = torch.stack(x_list, dim=0)               # [B, C, T, V]
    batched_mask = torch.stack(mask_list, dim=0)         # [B, 1, T, V]
    batched_labels = torch.stack(label_list, dim=0)      # [B]
    batched_gloss = torch.stack(gloss_list, dim=0)       # [B]
    batched_dec_in = torch.stack(dec_in_list, dim=0)     # [B, S]
    batched_dec_tgt = torch.stack(dec_tgt_list, dim=0)   # [B, S]
    batched_attn = torch.stack(attn_list, dim=0)         # [B, S]

    return {
        "x": batched_x,
        "mask": batched_mask,
        "label": batched_labels,
        "gloss_id": batched_gloss,
        "decoder_input": batched_dec_in,
        "decoder_target": batched_dec_tgt,
        "attention_mask": batched_attn,
        "metadata": meta_list
    }


def pad_variable_sequence_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collation function supporting variable-length temporal sequences:
    pads along the temporal dimension T to max(T) within the batch.
    """
    max_t = max(item["x"].shape[1] for item in batch)
    C = batch[0]["x"].shape[0]
    V = batch[0]["x"].shape[2]
    B = len(batch)

    padded_x = torch.zeros((B, C, max_t, V), dtype=torch.float32)
    padded_mask = torch.zeros((B, 1, max_t, V), dtype=torch.float32)

    for i, item in enumerate(batch):
        t_len = item["x"].shape[1]
        padded_x[i, :, :t_len, :] = item["x"]
        padded_mask[i, :, :t_len, :] = item["mask"]

    batched_labels = torch.stack([item["label"] for item in batch], dim=0)
    batched_gloss = torch.stack([item["gloss_id"] for item in batch], dim=0)
    batched_dec_in = torch.stack([item["decoder_input"] for item in batch], dim=0)
    batched_dec_tgt = torch.stack([item["decoder_target"] for item in batch], dim=0)
    batched_attn = torch.stack([item["attention_mask"] for item in batch], dim=0)
    meta_list = [item["metadata"] for item in batch]

    return {
        "x": padded_x,
        "mask": padded_mask,
        "label": batched_labels,
        "gloss_id": batched_gloss,
        "decoder_input": batched_dec_in,
        "decoder_target": batched_dec_tgt,
        "attention_mask": batched_attn,
        "metadata": meta_list
    }
