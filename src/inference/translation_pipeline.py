"""
SignTalk AI: End-to-End Translation Inference Pipeline.

Connects spatiotemporal landmark sequence input to natural language
English translation and Indian Sign Language gloss predictions:
  Landmarks [B, C, T, V]
      ↓
  ST-GCN + Transformer Model
      ↓
  Autoregressive Generation
      ↓
  Confidence Gating & Policy Check
      ↓
  Canonical Gloss & Natural English Translation
"""

from typing import Dict, Any, Optional, Union, List
import time
import numpy as np
import torch

from src.models.sign_translation_model import SignTranslationModel
from src.nlp.tokenizer import SignLanguageTokenizer


class SignTranslationPipeline:
    """
    Production-ready translation pipeline executing visual-to-linguistic inference.
    """

    def __init__(
        self,
        model: SignTranslationModel,
        tokenizer: SignLanguageTokenizer,
        confidence_threshold: float = 0.50,
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        Args:
            model: Instantiated and checkpoint-loaded SignTranslationModel.
            tokenizer: SignLanguageTokenizer instance.
            confidence_threshold: Minimum confidence score required to accept prediction.
            device: Target torch device ('cpu', 'cuda', etc.).
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.confidence_threshold = confidence_threshold

    @torch.no_grad()
    def translate_sequence(
        self,
        sequence: Union[torch.Tensor, np.ndarray],
        mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        max_length: int = 5
    ) -> Dict[str, Any]:
        """
        Translates a single sequence [C, T, V] or batch [1, C, T, V] into gloss and English.
        
        Args:
            sequence: Coordinates tensor or numpy array of shape [C, T, V] or [1, C, T, V].
            mask: Optional visibility mask [1, T, V] or [1, 1, T, V].
            max_length: Maximum decoding steps.
            
        Returns:
            Dictionary containing prediction strings, token IDs, confidence, status, and latency.
        """
        t0 = time.perf_counter()

        # Format input tensor to [1, C, T, V]
        if isinstance(sequence, np.ndarray):
            x = torch.from_numpy(sequence).float()
        else:
            x = sequence.clone().float()

        if x.dim() == 3:
            x = x.unsqueeze(0)  # [1, C, T, V]

        x = x.to(self.device)

        m = None
        if mask is not None:
            if isinstance(mask, np.ndarray):
                m = torch.from_numpy(mask).float()
            else:
                m = mask.clone().float()
            if m.dim() == 3:
                m = m.unsqueeze(0)
            m = m.to(self.device)

        t_prep = time.perf_counter()

        # Step 1: Feature Extraction
        t_vis_start = time.perf_counter()
        proj_features, visual_padding_mask = self.model.extract_visual_embeddings(x, m)
        t_vis_end = time.perf_counter()

        # Step 2: Autoregressive Decoding
        t_dec_start = time.perf_counter()
        token_ids_tensor, conf_tensor = self.model.transformer.generate(
            visual_features=proj_features,
            visual_padding_mask=visual_padding_mask,
            max_length=max_length
        )
        t_dec_end = time.perf_counter()

        # Format outputs
        token_ids = token_ids_tensor[0].cpu().tolist()
        confidences = conf_tensor[0].cpu().tolist()

        # Filter out <BOS>, <EOS>, <PAD> to extract primary lexical token
        lexical_tokens = []
        lexical_confs = []
        for tid, c in zip(token_ids, confidences):
            if tid not in {self.tokenizer.pad_id, self.tokenizer.bos_id, self.tokenizer.eos_id}:
                lexical_tokens.append(tid)
                lexical_confs.append(c)

        primary_token_id = lexical_tokens[0] if lexical_tokens else self.tokenizer.unk_id
        primary_confidence = float(lexical_confs[0]) if lexical_confs else 0.0

        # Confidence Gating Policy
        is_uncertain = primary_confidence < self.confidence_threshold
        if is_uncertain:
            gloss = "Uncertain sign"
            translation = "Please repeat"
            status = "REJECTED_LOW_CONFIDENCE"
        else:
            gloss = self.tokenizer.decode([primary_token_id], skip_special_tokens=True)
            translation = self.tokenizer.decode_to_translation([primary_token_id])
            status = "ACCEPTED"

        total_latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "predicted_gloss": gloss,
            "predicted_translation": translation,
            "token_ids": token_ids,
            "tokens": [self.tokenizer.id_to_token.get(t, "<UNK>") for t in token_ids],
            "primary_token_id": primary_token_id,
            "confidence": round(primary_confidence, 4),
            "is_uncertain": is_uncertain,
            "status": status,
            "latency": {
                "prep_ms": round((t_prep - t0) * 1000.0, 3),
                "visual_encoder_ms": round((t_vis_end - t_vis_start) * 1000.0, 3),
                "decoder_ms": round((t_dec_end - t_dec_start) * 1000.0, 3),
                "total_ms": round(total_latency_ms, 3)
            }
        }
