"""
SignTalk AI: Sign Transformer Alias Module.

Exposes SignTransformer as an alias to SignLanguageTransformer for architectural naming consistency.
"""

from src.models.sign_language_transformer import SignLanguageTransformer

# Alias matching prompt specification
SignTransformer = SignLanguageTransformer

__all__ = ["SignTransformer", "SignLanguageTransformer"]
