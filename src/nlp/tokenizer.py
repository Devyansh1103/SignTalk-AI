"""
SignTalk AI: Sign Language Tokenizer.

Handles tokenization, detokenization, vocabulary indexing, and translation
lookups for sign language glosses and target translation phrases.
"""

from typing import List, Dict, Any, Optional, Union
import os
import json
import torch

from src.nlp.text_normalizer import TextNormalizer


class SignLanguageTokenizer:
    """
    Tokenizer for Indian Sign Language glosses and translation tokens.
    """

    def __init__(
        self,
        vocab_path: str = "assets/vocabularies/mvp_10.json",
        normalizer: Optional[TextNormalizer] = None
    ):
        """
        Args:
            vocab_path: Path to vocabulary JSON configuration file.
            normalizer: Optional TextNormalizer instance.
        """
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_path = vocab_path
        self.vocab_name = data.get("vocabulary_name", "custom")
        self.vocab_version = data.get("version", "1.0.0")

        self.special_tokens = data.get("special_tokens", {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3
        })
        self.pad_id = self.special_tokens.get("<PAD>", 0)
        self.unk_id = self.special_tokens.get("<UNK>", 1)
        self.bos_id = self.special_tokens.get("<BOS>", 2)
        self.eos_id = self.special_tokens.get("<EOS>", 3)

        self.token_to_id: Dict[str, int] = {k: int(v) for k, v in data["token_to_id"].items()}
        self.id_to_token: Dict[int, str] = {int(k): str(v) for k, v in data["id_to_token"].items()}

        # Translation mappings if present in vocabulary file
        self.class_id_to_token_id = {int(k): int(v) for k, v in data.get("class_id_to_token_id", {}).items()}
        self.token_id_to_class_id = {v: k for k, v in self.class_id_to_token_id.items()}
        self.class_id_to_translation = {int(k): str(v) for k, v in data.get("class_id_to_translation", {}).items()}
        self.class_id_to_gloss = {int(k): str(v) for k, v in data.get("class_id_to_gloss", {}).items()}

        self.normalizer = normalizer if normalizer is not None else TextNormalizer()

    @property
    def pad_token_id(self) -> int:
        return self.pad_id

    @property
    def bos_token_id(self) -> int:
        return self.bos_id

    @property
    def eos_token_id(self) -> int:
        return self.eos_id

    @property
    def unk_token_id(self) -> int:
        return self.unk_id

    @property
    def vocab_size(self) -> int:
        """Returns total vocabulary size including special tokens."""
        return len(self.token_to_id)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        pad_to_max: bool = False
    ) -> List[int]:
        """
        Encodes a single gloss string or word into a list of token IDs.
        
        Args:
            text: Input gloss or text string (e.g., 'HELLO' or 'THANK_YOU').
            add_special_tokens: If True, wraps sequence with [<BOS>, ..., <EOS>].
            max_length: Optional maximum sequence length constraint.
            pad_to_max: If True, pads sequence up to max_length with pad_id.
            
        Returns:
            List of integer token IDs.
        """
        clean_text = self.normalizer.normalize_gloss(text)
        
        # Look up token; fallback to UNK
        token_id = self.token_to_id.get(clean_text, self.unk_id)
        tokens = [token_id]

        if add_special_tokens:
            tokens = [self.bos_id] + tokens + [self.eos_id]

        if max_length is not None:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            elif pad_to_max:
                tokens = tokens + [self.pad_id] * (max_length - len(tokens))

        return tokens

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decodes a list of token IDs back into a canonical gloss string.
        
        Args:
            token_ids: Sequence of integer token IDs.
            skip_special_tokens: If True, removes <PAD>, <UNK>, <BOS>, <EOS>.
            
        Returns:
            Decoded string.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()

        special_set = {self.pad_id, self.unk_id, self.bos_id, self.eos_id} if skip_special_tokens else set()
        words = []
        for tid in token_ids:
            if skip_special_tokens and tid in special_set:
                continue
            word = self.id_to_token.get(int(tid), "<UNK>")
            words.append(word)

        return " ".join(words).strip()

    def decode_to_translation(
        self,
        token_ids: Union[List[int], torch.Tensor]
    ) -> str:
        """
        Decodes token sequence into natural language English translation.
        
        Args:
            token_ids: Sequence of integer token IDs.
            
        Returns:
            English translation string.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()

        # Find first non-special token
        for tid in token_ids:
            if tid in self.token_id_to_class_id:
                cid = self.token_id_to_class_id[tid]
                return self.class_id_to_translation.get(cid, self.id_to_token.get(tid, "Uncertain"))

        return "Uncertain sign"

    def class_id_to_token(self, class_id: int) -> int:
        """Maps a 0-indexed categorical class ID to its vocabulary token ID."""
        if class_id not in self.class_id_to_token_id:
            raise KeyError(f"Class ID {class_id} not mapped in vocabulary.")
        return self.class_id_to_token_id[class_id]

    def token_to_class_id(self, token_id: int) -> Optional[int]:
        """Maps a vocabulary token ID back to 0-indexed class ID."""
        return self.token_id_to_class_id.get(token_id, None)
