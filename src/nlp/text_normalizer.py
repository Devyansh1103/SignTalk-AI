"""
SignTalk AI: Text Normalization Module.

Provides deterministic string cleaning, Unicode normalization, whitespace
collapsing, and token-level harmonization for sign language glosses and
natural language English translations.
"""

import unicodedata
import re
from typing import Optional


class TextNormalizer:
    """
    Deterministic text normalizer for sign language glosses and translations.
    """

    def __init__(
        self,
        lowercase: bool = False,
        strip_punctuation: bool = False,
        normalize_unicode: str = "NFKC"
    ):
        """
        Args:
            lowercase: If True, lowercases the normalized text.
            strip_punctuation: If True, strips standard punctuation.
            normalize_unicode: Unicode normalization form ('NFKC', 'NFC', 'NFD', 'NFKD').
        """
        self.lowercase = lowercase
        self.strip_punctuation = strip_punctuation
        self.normalize_unicode = normalize_unicode

    def normalize(self, text: str) -> str:
        """
        Applies sequential normalization passes:
          1. Unicode normalization (default NFKC)
          2. Whitespace collapse and trimming
          3. Optional punctuation removal (preserving alphanumerics and basic hyphens)
          4. Optional lowercasing
        """
        if not isinstance(text, str):
            text = str(text)

        # 1. Unicode decomposition / canonical composition
        if self.normalize_unicode:
            text = unicodedata.normalize(self.normalize_unicode, text)

        # 2. Replace non-standard whitespace / tabs / newlines with single space
        text = re.sub(r"\s+", " ", text).strip()

        # 3. Optional punctuation filtering
        if self.strip_punctuation:
            # Preserve letters, numbers, spaces, and hyphens/underscores
            text = re.sub(r"[^\w\s\-_]", "", text)

        # 4. Casing policy
        if self.lowercase:
            text = text.lower()

        return text

    def normalize_gloss(self, gloss: str) -> str:
        """
        Normalizes a sign language gloss:
          - Unicode NFKC
          - Upper-case
          - Replaces spaces with underscores if multi-token
        """
        norm = self.normalize(gloss)
        norm = norm.upper()
        norm = re.sub(r"\s+", "_", norm)
        return norm

    def normalize_translation(self, translation: str) -> str:
        """
        Normalizes an English translation sentence or phrase:
          - Unicode NFKC
          - Strips outer whitespace
          - Ensures consistent single spacing
        """
        return self.normalize(translation)
