"""
SignTalk AI: Sign Language Transformer Module.

Implements an encoder-decoder Transformer architecture for sign language
gloss/token sequence modeling:
  - Visual Feature Encoder: Models temporal dynamics across ST-GCN sequence representations
  - Language Decoder: Autoregressively generates target gloss / translation tokens with cross-attention
  - Supports causal masking, padding masks, and greedy / beam search generation
"""

from typing import Optional, Dict, Any, List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers.positional_encoding import SinusoidalPositionalEncoding
from src.models.transformer_masks import generate_causal_bool_mask, create_padding_mask


class SignLanguageTransformer(nn.Module):
    """
    Encoder-Decoder Transformer for translating spatiotemporal visual representations
    into symbolic linguistic tokens (gloss sequences / translation words).
    """

    def __init__(
        self,
        vocab_size: int = 14,
        embed_dim: int = 128,
        num_heads: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.1,
        max_visual_len: int = 64,
        max_target_len: int = 32,
        pad_idx: int = 0,
        bos_idx: int = 2,
        eos_idx: int = 3
    ):
        """
        Args:
            vocab_size: Size of token vocabulary including special tokens.
            embed_dim: Hidden representation dimension (d_model).
            num_heads: Number of multi-head attention heads (must divide embed_dim).
            encoder_layers: Number of Transformer encoder layers.
            decoder_layers: Number of Transformer decoder layers.
            ff_dim: Dimension of feed-forward network in Transformer layers.
            dropout: Dropout probability.
            max_visual_len: Maximum visual sequence length.
            max_target_len: Maximum target token sequence length.
            pad_idx: Token ID of <PAD>.
            bos_idx: Token ID of <BOS>.
            eos_idx: Token ID of <EOS>.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        # 1. Target Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # 2. Positional Encodings
        self.visual_pos_encoder = SinusoidalPositionalEncoding(
            embed_dim=embed_dim, dropout=dropout, max_len=max_visual_len
        )
        self.target_pos_encoder = SinusoidalPositionalEncoding(
            embed_dim=embed_dim, dropout=dropout, max_len=max_target_len
        )

        # 3. Transformer Encoder
        encoder_norm = nn.LayerNorm(embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=encoder_layers,
            norm=encoder_norm,
            enable_nested_tensor=False
        )

        # 4. Transformer Decoder
        decoder_norm = nn.LayerNorm(embed_dim)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer,
            num_layers=decoder_layers,
            norm=decoder_norm
        )

        # 5. Output Projection Head to Token Vocabulary
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initializes weights using Xavier normal distribution."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if self.pad_idx is not None and self.pad_idx < self.vocab_size:
            with torch.no_grad():
                self.token_embedding.weight[self.pad_idx].fill_(0.0)

        nn.init.xavier_normal_(self.lm_head.weight)
        if self.lm_head.bias is not None:
            nn.init.constant_(self.lm_head.bias, 0.0)

    def encode(
        self,
        visual_features: torch.Tensor,
        visual_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encodes projected visual feature sequence into latent spatiotemporal memory.
        
        Args:
            visual_features: Tensor of shape [B, T_visual, embed_dim].
            visual_padding_mask: Boolean tensor of shape [B, T_visual] where True indicates padded frames.
            
        Returns:
            Memory tensor of shape [B, T_visual, embed_dim].
        """
        x = self.visual_pos_encoder(visual_features)
        memory = self.encoder(x, src_key_padding_mask=visual_padding_mask)
        return memory

    def decode(
        self,
        target_tokens: torch.Tensor,
        memory: torch.Tensor,
        visual_padding_mask: Optional[torch.Tensor] = None,
        target_padding_mask: Optional[torch.Tensor] = None,
        target_causal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decodes target token sequence conditioned on visual memory.
        
        Args:
            target_tokens: Long tensor of shape [B, S].
            memory: Visual encoder output [B, T_visual, embed_dim].
            visual_padding_mask: Boolean tensor [B, T_visual] indicating padded visual frames.
            target_padding_mask: Boolean tensor [B, S] indicating padded tokens.
            target_causal_mask: Boolean upper-triangular mask [S, S].
            
        Returns:
            Decoder hidden states [B, S, embed_dim].
        """
        B, S = target_tokens.shape
        device = target_tokens.device

        # Scale embeddings by sqrt(d_model) as in standard Transformer
        tgt_emb = self.token_embedding(target_tokens) * math.sqrt(self.embed_dim)
        tgt_emb = self.target_pos_encoder(tgt_emb)

        if target_causal_mask is None:
            target_causal_mask = generate_causal_bool_mask(S, device=device)

        if target_padding_mask is None:
            target_padding_mask = create_padding_mask(target_tokens, pad_idx=self.pad_idx)

        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=target_causal_mask,
            memory_mask=None,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=visual_padding_mask
        )
        return out

    def forward(
        self,
        visual_features: torch.Tensor,
        target_tokens: torch.Tensor,
        visual_padding_mask: Optional[torch.Tensor] = None,
        target_padding_mask: Optional[torch.Tensor] = None,
        target_causal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Full forward pass for teacher-forced training.
        
        Args:
            visual_features: Projected visual sequence [B, T_visual, embed_dim].
            target_tokens: Decoder input token IDs [B, S].
            visual_padding_mask: Optional padding mask [B, T_visual].
            target_padding_mask: Optional target padding mask [B, S].
            target_causal_mask: Optional causal subsequent mask [S, S].
            
        Returns:
            Vocabulary logits of shape [B, S, vocab_size].
        """
        memory = self.encode(visual_features, visual_padding_mask=visual_padding_mask)
        dec_hidden = self.decode(
            target_tokens=target_tokens,
            memory=memory,
            visual_padding_mask=visual_padding_mask,
            target_padding_mask=target_padding_mask,
            target_causal_mask=target_causal_mask
        )
        logits = self.lm_head(dec_hidden)
        return logits

    @torch.no_grad()
    def generate(
        self,
        visual_features: torch.Tensor,
        visual_padding_mask: Optional[torch.Tensor] = None,
        max_length: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy autoregressive decoding.
        
        Args:
            visual_features: Projected visual sequence [B, T_visual, embed_dim].
            visual_padding_mask: Optional padding mask [B, T_visual].
            max_length: Maximum generation steps.
            
        Returns:
            Tuple of (generated_token_ids [B, S], confidence_scores [B, S]).
        """
        self.eval()
        B = visual_features.shape[0]
        device = visual_features.device

        memory = self.encode(visual_features, visual_padding_mask=visual_padding_mask)

        # Initialize sequence with <BOS>
        current_tokens = torch.full((B, 1), self.bos_idx, dtype=torch.long, device=device)
        confidences = torch.ones((B, 1), dtype=torch.float32, device=device)

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_length - 1):
            dec_hidden = self.decode(
                target_tokens=current_tokens,
                memory=memory,
                visual_padding_mask=visual_padding_mask
            )
            # Logits for last position
            next_logits = self.lm_head(dec_hidden[:, -1, :])  # [B, vocab_size]
            probs = F.softmax(next_logits, dim=-1)

            top_prob, next_token = probs.max(dim=-1)  # [B]

            # Replace token with <PAD> if sample already emitted <EOS>
            next_token_masked = torch.where(finished, torch.tensor(self.pad_idx, device=device), next_token)
            next_prob_masked = torch.where(finished, torch.tensor(1.0, device=device), top_prob)

            current_tokens = torch.cat([current_tokens, next_token_masked.unsqueeze(1)], dim=1)
            confidences = torch.cat([confidences, next_prob_masked.unsqueeze(1)], dim=1)

            finished = finished | (next_token == self.eos_idx)
            if finished.all():
                break

        return current_tokens, confidences
