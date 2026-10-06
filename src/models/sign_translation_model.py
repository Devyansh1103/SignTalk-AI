"""
SignTalk AI: End-to-End Sign Translation Model.

Couples the Spatial-Temporal Graph Convolutional Network (ST-GCN) visual
encoder with the autoregressive SignLanguageTransformer decoder:
  Landmarks [B, C, T, V]
      ↓
  ST-GCN Spatiotemporal Encoder
      ↓
  Temporal Sequence Projection [B, T', D_model]
      ↓
  Transformer Encoder-Decoder
      ↓
  Token Logits [B, S, V_vocab] / Autoregressively Generated Tokens
"""

from typing import Optional, Dict, Any, Tuple, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.stgcn import SignSTGCN
from src.models.layers.feature_projection import FeatureProjection
from src.models.sign_language_transformer import SignLanguageTransformer
from src.models.transformer_masks import downsample_temporal_mask


class SignTranslationModel(nn.Module):
    """
    End-to-End Translation Architecture integrating ST-GCN with Transformer NLP layers.
    """

    def __init__(
        self,
        stgcn_config: Optional[Dict[str, Any]] = None,
        transformer_config: Optional[Dict[str, Any]] = None,
        freeze_stgcn: bool = True,
        stgcn_checkpoint: Optional[str] = None
    ):
        """
        Args:
            stgcn_config: Hyperparameters for SignSTGCN module.
            transformer_config: Hyperparameters for SignLanguageTransformer and FeatureProjection.
            freeze_stgcn: If True, freezes ST-GCN visual encoder parameters.
            stgcn_checkpoint: Optional file path to pretrained ST-GCN weights.
        """
        super().__init__()

        # 1. ST-GCN Visual Encoder
        stgcn_cfg = stgcn_config or {}
        self.stgcn = SignSTGCN(
            in_channels=stgcn_cfg.get("in_channels", 3),
            num_classes=stgcn_cfg.get("num_classes", 10),
            num_nodes=stgcn_cfg.get("num_nodes", 93),
            sequence_length=stgcn_cfg.get("sequence_length", 45),
            graph_strategy=stgcn_cfg.get("graph_strategy", "spatial"),
            block_channels=stgcn_cfg.get("block_channels", [64, 64, 128, 128, 256, 256]),
            block_strides=stgcn_cfg.get("block_strides", [1, 1, 2, 1, 2, 1]),
            temporal_kernel_size=stgcn_cfg.get("temporal_kernel_size", 9),
            dropout=stgcn_cfg.get("dropout", 0.3),
            residual=stgcn_cfg.get("residual", True),
            use_learnable_edge_weights=stgcn_cfg.get("use_learnable_edge_weights", True)
        )

        stgcn_feat_dim = self.stgcn.block_channels[-1]  # 256

        # 2. Transformer Configuration
        tf_cfg = transformer_config or {}
        embed_dim = tf_cfg.get("embed_dim", 128)
        vocab_size = tf_cfg.get("vocab_size", 14)
        num_heads = tf_cfg.get("num_heads", 4)
        encoder_layers = tf_cfg.get("encoder_layers", 2)
        decoder_layers = tf_cfg.get("decoder_layers", 2)
        ff_dim = tf_cfg.get("ff_dim", 512)
        dropout = tf_cfg.get("dropout", 0.1)
        max_visual_len = tf_cfg.get("max_visual_len", 64)
        max_target_len = tf_cfg.get("max_target_len", 32)
        pad_idx = tf_cfg.get("pad_idx", 0)
        bos_idx = tf_cfg.get("bos_idx", 2)
        eos_idx = tf_cfg.get("eos_idx", 3)

        # 3. Feature Projection Layer (ST-GCN channels -> Transformer embed_dim)
        self.feature_projection = FeatureProjection(
            in_dim=stgcn_feat_dim,
            embed_dim=embed_dim,
            dropout=dropout,
            use_layer_norm=True
        )

        # 4. Sign Language Transformer
        self.transformer = SignLanguageTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            max_visual_len=max_visual_len,
            max_target_len=max_target_len,
            pad_idx=pad_idx,
            bos_idx=bos_idx,
            eos_idx=eos_idx
        )

        self.freeze_stgcn_flag = freeze_stgcn

        # 5. Load Pretrained ST-GCN Checkpoint if provided
        if stgcn_checkpoint is not None and os.path.exists(stgcn_checkpoint):
            self.load_stgcn_checkpoint(stgcn_checkpoint)

        # Apply freezing
        if freeze_stgcn:
            self.freeze_stgcn(True)

    def load_stgcn_checkpoint(self, checkpoint_path: str):
        """
        Loads pretrained weights into the ST-GCN visual encoder.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"ST-GCN checkpoint not found at: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Load weights into stgcn module
        missing, unexpected = self.stgcn.load_state_dict(state_dict, strict=False)
        print(f"[SignTranslationModel] Loaded ST-GCN checkpoint from {checkpoint_path}")
        if missing:
            print(f"  Missing keys: {len(missing)} (expected if classification head is unused)")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

    def freeze_stgcn(self, freeze: bool = True):
        """
        Controls freezing of the ST-GCN visual encoder parameters.
        """
        self.freeze_stgcn_flag = freeze
        for param in self.stgcn.parameters():
            param.requires_grad = not freeze
        if freeze:
            self.stgcn.eval()

    def train(self, mode: bool = True):
        """
        Custom train mode: keeps ST-GCN in eval mode if frozen.
        """
        super().train(mode)
        if self.freeze_stgcn_flag:
            self.stgcn.eval()
        return self

    def extract_visual_embeddings(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extracts and projects visual spatiotemporal features from landmarks.
        
        Args:
            x: Input landmark tensor [B, C, T, V].
            mask: Optional visibility mask [B, 1, T, V].
            
        Returns:
            Tuple of:
              - Projected feature sequence [B, T', D_model]
              - Downsampled visual padding mask [B, T'] (or None)
        """
        # If ST-GCN is frozen, do not compute gradients through it
        if self.freeze_stgcn_flag:
            with torch.no_grad():
                seq_features = self.stgcn.extract_temporal_sequence(x, mask)
        else:
            seq_features = self.stgcn.extract_temporal_sequence(x, mask)

        # [B, T'=12, C_final=256] -> [B, 12, embed_dim]
        proj_features = self.feature_projection(seq_features)

        # Downsample visual mask if provided
        visual_padding_mask = None
        if mask is not None:
            T_prime = proj_features.shape[1]
            visual_padding_mask = downsample_temporal_mask(mask, target_len=T_prime)

        return proj_features, visual_padding_mask

    def forward(
        self,
        x: torch.Tensor,
        target_tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target_padding_mask: Optional[torch.Tensor] = None,
        target_causal_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Full teacher-forced forward pass.
        
        Args:
            x: Input landmarks [B, C, T, V].
            target_tokens: Decoder input tokens [B, S].
            mask: Optional landmark validity mask [B, 1, T, V].
            target_padding_mask: Optional boolean padding mask [B, S].
            target_causal_mask: Optional subsequent causal mask [S, S].
            
        Returns:
            Vocabulary logits tensor of shape [B, S, vocab_size].
        """
        proj_features, visual_padding_mask = self.extract_visual_embeddings(x, mask)
        logits = self.transformer(
            visual_features=proj_features,
            target_tokens=target_tokens,
            visual_padding_mask=visual_padding_mask,
            target_padding_mask=target_padding_mask,
            target_causal_mask=target_causal_mask
        )
        return logits

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        max_length: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive generation from input landmark sequence.
        
        Args:
            x: Input landmarks [B, C, T, V].
            mask: Optional visibility mask [B, 1, T, V].
            max_length: Maximum decoding steps.
            
        Returns:
            Tuple of:
              - generated token IDs [B, S]
              - confidence probability scores [B, S]
        """
        self.eval()
        proj_features, visual_padding_mask = self.extract_visual_embeddings(x, mask)
        return self.transformer.generate(
            visual_features=proj_features,
            visual_padding_mask=visual_padding_mask,
            max_length=max_length
        )
