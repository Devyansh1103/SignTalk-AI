"""
SignTalk AI - Real-Time ST-GCN Model Runner
Loads the certified ST-GCN checkpoint, executes tensor shape validation,
performs warm-up, and runs optimized evaluation forward passes to produce
raw logits, softmax probabilities, and structured PredictionResult objects.
"""

import os
import time
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn

from src.models.stgcn import SignSTGCN
from src.data.node_schema import TOTAL_NODES, CHANNELS, TARGET_SEQUENCE_LENGTH
from src.data.label_map import (
    NUM_CLASSES,
    get_label,
    get_gloss,
    get_translation
)
from src.realtime.types import LandmarkFrame, PredictionResult

logger = logging.getLogger("SignTalk.RealTime.ModelRunner")


class ModelRunnerError(Exception):
    """Base exception for model execution errors."""
    pass


class ShapeValidationError(ModelRunnerError):
    """Raised when input tensor does not strictly match [1, 3, 45, 93]."""
    pass


class STGCNRunner:
    """
    Production inference runner for SignTalk_STGCN_v1.
    """

    def __init__(
        self,
        checkpoint_path: str = "experiments/stgcn/checkpoints/best_checkpoint.pt",
        device: str = "auto",
        top_k: int = 3,
        warmup_iterations: int = 5
    ):
        """
        Args:
            checkpoint_path: Path to PyTorch model checkpoint (.pt).
            device: Target execution device ('auto', 'cpu', 'cuda').
            top_k: Number of ranked predictions to return (default: 3).
            warmup_iterations: Number of dummy inference cycles during initialization.
        """
        self.checkpoint_path = checkpoint_path
        self.top_k = max(1, min(NUM_CLASSES, int(top_k)))
        self.device = self._resolve_device(device)

        self.expected_channels = CHANNELS                # 3
        self.expected_sequence_length = TARGET_SEQUENCE_LENGTH  # 45
        self.expected_nodes = TOTAL_NODES                # 93
        self.expected_num_classes = NUM_CLASSES          # 10

        self.model: Optional[SignSTGCN] = None
        self._load_model()

        if warmup_iterations > 0:
            self.warmup(num_iterations=warmup_iterations)

    def _resolve_device(self, req_device: str) -> torch.device:
        """Resolves target torch device with strict error handling."""
        req_clean = req_device.strip().lower()
        if req_clean == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif req_clean == "cuda":
            if not torch.cuda.is_available():
                raise ModelRunnerError(
                    "CUDA device requested ('--device cuda'), but CUDA is not available on this machine."
                )
            return torch.device("cuda")
        elif req_clean == "cpu":
            return torch.device("cpu")
        else:
            try:
                dev = torch.device(req_clean)
                return dev
            except Exception as e:
                raise ModelRunnerError(f"Invalid device specification '{req_device}': {e}")

    def _load_model(self) -> None:
        """Instantiates SignSTGCN and loads weights from checkpoint."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"ST-GCN checkpoint not found at: {self.checkpoint_path}. "
                "Ensure Phase 3 ST-GCN training has completed."
            )

        logger.info(f"Loading ST-GCN model from checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

        # Extract model configuration from checkpoint
        cfg = checkpoint.get("config", {})
        model_kwargs = cfg.get("model", {})

        # Default fallback parameters if config dict is partial
        self.model = SignSTGCN(
            in_channels=self.expected_channels,
            num_classes=self.expected_num_classes,
            num_nodes=self.expected_nodes,
            sequence_length=self.expected_sequence_length,
            graph_strategy=cfg.get("graph", {}).get("strategy", "spatial"),
            block_channels=model_kwargs.get("block_channels", [64, 64, 128, 128, 256, 256]),
            block_strides=model_kwargs.get("block_strides", [1, 1, 2, 1, 2, 1]),
            temporal_kernel_size=model_kwargs.get("temporal_kernel_size", 9),
            dropout=model_kwargs.get("dropout", 0.3),
            residual=model_kwargs.get("residual", True),
            use_learnable_edge_weights=model_kwargs.get("use_learnable_edge_weights", True)
        )

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Disable gradients permanently for inference
        for param in self.model.parameters():
            param.requires_grad = False

        logger.info(
            f"Successfully initialized SignSTGCN on {self.device} "
            f"(Epoch: {checkpoint.get('epoch', 'N/A')}, Parameters: {sum(p.numel() for p in self.model.parameters()):,})"
        )

    def warmup(self, num_iterations: int = 5) -> None:
        """Executes dummy inference forward passes to prime GPU kernels and memory."""
        logger.info(f"Warming up ST-GCN model ({num_iterations} cycles on {self.device})...")
        dummy_x = torch.zeros(
            (1, self.expected_channels, self.expected_sequence_length, self.expected_nodes),
            dtype=torch.float32,
            device=self.device
        )
        dummy_mask = torch.ones(
            (1, 1, self.expected_sequence_length, self.expected_nodes),
            dtype=torch.float32,
            device=self.device
        )

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(dummy_x, dummy_mask)

        logger.info("Warm-up complete.")

    def window_to_tensor(
        self,
        window: List[LandmarkFrame]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts a list of T LandmarkFrame objects into ST-GCN input tensors.
        
        Args:
            window: List of LandmarkFrame objects of length T.

        Returns:
            Tuple of:
              - x: torch.FloatTensor of shape [1, 3, T, 93] on self.device
              - mask: torch.FloatTensor of shape [1, 1, T, 93] on self.device
        """
        T = len(window)
        if T != self.expected_sequence_length:
            raise ShapeValidationError(
                f"Window length mismatch: got T={T} frames, expected exactly {self.expected_sequence_length}"
            )

        # Preallocate numpy arrays: (T, 93, 3) and (T, 93)
        coords_array = np.zeros((T, self.expected_nodes, self.expected_channels), dtype=np.float32)
        mask_array = np.zeros((T, self.expected_nodes), dtype=bool)

        for t_idx, lf in enumerate(window):
            if lf.normalized_coords.shape != (self.expected_nodes, self.expected_channels):
                raise ShapeValidationError(
                    f"Frame {lf.frame_id} has invalid normalized_coords shape: "
                    f"{lf.normalized_coords.shape}, expected ({self.expected_nodes}, {self.expected_channels})"
                )
            coords_array[t_idx] = lf.normalized_coords
            mask_array[t_idx] = lf.mask

        # Numerical safety check
        if np.isnan(coords_array).any() or np.isinf(coords_array).any():
            raise ModelRunnerError("NaN or Infinite values detected in window landmark tensor")

        # Reorder dimensions: (T, V, C) -> (C, T, V) -> (1, C, T, V)
        tensor_c_t_v = np.transpose(coords_array, (2, 0, 1))  # (3, T, 93)
        tensor_x = torch.from_numpy(tensor_c_t_v).float().unsqueeze(0)  # [1, 3, T, 93]

        # Mask dimension: (T, V) -> (1, 1, T, V)
        tensor_mask = torch.from_numpy(mask_array).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T, 93]

        # Final assertion check
        expected_shape = (1, self.expected_channels, self.expected_sequence_length, self.expected_nodes)
        if tensor_x.shape != expected_shape:
            raise ShapeValidationError(
                f"Constructed input shape {tensor_x.shape} does not match expected {expected_shape}"
            )

        return tensor_x.to(self.device), tensor_mask.to(self.device)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes raw model forward pass.
        
        Args:
            x: Input tensor [1, 3, 45, 93]
            mask: Optional visibility mask [1, 1, 45, 93]
            
        Returns:
            Tuple of:
              - logits: torch.FloatTensor of shape [1, 10]
              - probabilities: torch.FloatTensor of shape [1, 10]
        """
        if x.dim() != 4 or x.shape[1] != self.expected_channels or x.shape[2] != self.expected_sequence_length or x.shape[3] != self.expected_nodes:
            raise ShapeValidationError(
                f"Invalid input tensor shape {x.shape}. Expected [1, {self.expected_channels}, {self.expected_sequence_length}, {self.expected_nodes}]"
            )

        logits = self.model(x, mask)
        probabilities = torch.softmax(logits, dim=-1)
        return logits, probabilities

    def predict_window(
        self,
        window: List[LandmarkFrame]
    ) -> PredictionResult:
        """
        End-to-end inference on a sliding window of LandmarkFrame instances.
        
        Returns:
            Structured PredictionResult containing class label, confidence,
            top-k alternatives, and timing diagnostics.
        """
        if not window:
            raise ValueError("Cannot predict on empty window")

        window_start_time = window[0].timestamp
        window_end_time = window[-1].timestamp
        window_frame_count = len(window)

        # Compute average input quality across window
        qualities = [f.quality.quality_score for f in window]
        input_quality = float(np.mean(qualities)) if qualities else 0.0
        valid_frames = sum(1 for f in window if f.quality.is_valid)
        is_valid_quality = (valid_frames >= 15)

        # Tensor preparation
        x, mask = self.window_to_tensor(window)

        # Execute timed forward pass
        t_start = time.perf_counter()
        logits_t, probs_t = self.predict(x, mask)
        t_end = time.perf_counter()

        inference_latency_ms = (t_end - t_start) * 1000.0

        logits_np = logits_t[0].cpu().numpy()
        probs_np = probs_t[0].cpu().numpy()

        pred_class_id = int(np.argmax(probs_np))
        confidence = float(probs_np[pred_class_id])

        # Top-K prediction calculation
        top_indices = np.argsort(probs_np)[::-1][:self.top_k]
        top_k_list = [
            (int(idx), get_label(int(idx)), float(probs_np[idx]))
            for idx in top_indices
        ]

        label = get_label(pred_class_id)
        gloss = get_gloss(pred_class_id)
        translation = get_translation(pred_class_id)

        return PredictionResult(
            class_id=pred_class_id,
            label=label,
            gloss=gloss,
            translation=translation,
            confidence=confidence,
            probabilities=probs_np,
            logits=logits_np,
            top_k=top_k_list,
            timestamp=time.time(),
            window_start_time=window_start_time,
            window_end_time=window_end_time,
            window_frame_count=window_frame_count,
            inference_start_time=t_start,
            inference_end_time=t_end,
            inference_latency_ms=round(inference_latency_ms, 2),
            input_quality=round(input_quality, 4),
            is_valid_quality=is_valid_quality,
            buffer_length=window_frame_count,
            device=str(self.device)
        )
