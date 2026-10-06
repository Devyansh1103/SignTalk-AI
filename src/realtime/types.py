"""
SignTalk AI - Real-Time Data Types and Contract Classes
Defines typed structures for frame packets, multimodal landmark frames,
quality assessments, and streaming performance metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch

from src.data.node_schema import TOTAL_NODES, CHANNELS


@dataclass
class FramePacket:
    """
    Encapsulates a single video frame captured from the camera or video source.
    """
    frame_id: int
    timestamp: float          # Monotonic timestamp via time.perf_counter()
    capture_time: float       # Wall-clock timestamp via time.time()
    image: np.ndarray         # Image array (H, W, 3) in BGR or RGB
    width: int
    height: int
    is_rgb: bool = False
    is_flipped: bool = False


@dataclass
class ModalityDetection:
    """
    Status and confidence metrics for individual sensory modalities in a single frame.
    """
    pose_detected: bool = False
    left_hand_detected: bool = False
    right_hand_detected: bool = False
    face_detected: bool = False
    pose_confidence: float = 0.0
    left_hand_confidence: float = 0.0
    right_hand_confidence: float = 0.0
    face_confidence: float = 0.0

    @property
    def either_hand_detected(self) -> bool:
        return self.left_hand_detected or self.right_hand_detected

    @property
    def both_hands_detected(self) -> bool:
        return self.left_hand_detected and self.right_hand_detected

    @property
    def dominant_hand_confidence(self) -> float:
        return max(self.left_hand_confidence, self.right_hand_confidence)


@dataclass
class QualityReport:
    """
    Real-time quality validation report for a single landmark frame.
    """
    is_valid: bool
    quality_score: float
    classification: str       # 'GOOD', 'ACCEPTABLE', 'REVIEW', 'REJECT'
    missing_ratio: float      # Ratio of unobserved nodes in 0..92 [0.0, 1.0]
    has_nan_or_inf: bool
    coordinate_in_range: bool
    sudden_jumps_detected: bool
    max_jump_distance: float = 0.0
    rejection_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class LandmarkFrame:
    """
    Canonical 93-node landmark representation for a single real-time frame.
    Guaranteed to conform to the 93-node-v1 schema with 3 spatial channels.
    """
    frame_id: int
    timestamp: float                    # Monotonic timestamp
    raw_coords: np.ndarray              # Shape: (93, 3) in raw MediaPipe space
    normalized_coords: np.ndarray       # Shape: (93, 3) in torso-centered space
    mask: np.ndarray                    # Shape: (93,) boolean detection mask
    visibility: np.ndarray              # Shape: (93,) float confidence/visibility
    modality_stats: ModalityDetection
    quality: QualityReport

    # Latency instrumentation (milliseconds)
    capture_latency_ms: float = 0.0
    preprocessing_latency_ms: float = 0.0
    extraction_latency_ms: float = 0.0
    normalization_latency_ms: float = 0.0
    quality_check_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    def __post_init__(self):
        # Validate shapes and numerical safety
        if self.raw_coords.shape != (TOTAL_NODES, CHANNELS):
            raise ValueError(f"raw_coords shape must be ({TOTAL_NODES}, {CHANNELS}), got {self.raw_coords.shape}")
        if self.normalized_coords.shape != (TOTAL_NODES, CHANNELS):
            raise ValueError(f"normalized_coords shape must be ({TOTAL_NODES}, {CHANNELS}), got {self.normalized_coords.shape}")
        if self.mask.shape != (TOTAL_NODES,):
            raise ValueError(f"mask shape must be ({TOTAL_NODES},), got {self.mask.shape}")
        if self.visibility.shape != (TOTAL_NODES,):
            raise ValueError(f"visibility shape must be ({TOTAL_NODES},), got {self.visibility.shape}")

    def to_stgcn_frame_tensor(self, include_batch: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Formats single frame into ST-GCN shape:
          Single frame: data=[C, 1, V] = [3, 1, 93], mask=[1, 1, 93]
          Batched: data=[1, 3, 1, 93], mask=[1, 1, 1, 93]
        """
        # normalized_coords is (93, 3) -> transpose to (3, 1, 93)
        c_v = self.normalized_coords.T  # (3, 93)
        c_1_v = c_v[:, np.newaxis, :]   # (3, 1, 93)
        m_1_v = self.mask[np.newaxis, np.newaxis, :].astype(np.float32)  # (1, 1, 93)

        tensor_x = torch.from_numpy(c_1_v.astype(np.float32))
        tensor_m = torch.from_numpy(m_1_v)

        if include_batch:
            tensor_x = tensor_x.unsqueeze(0)  # [1, 3, 1, 93]
            tensor_m = tensor_m.unsqueeze(0)  # [1, 1, 1, 93]

        return tensor_x, tensor_m


@dataclass
class StreamMetrics:
    """
    Rolling performance, throughput, and latency statistics for the landmark stream.
    """
    total_captured_frames: int = 0
    total_processed_frames: int = 0
    total_dropped_frames: int = 0
    capture_fps: float = 0.0
    processing_fps: float = 0.0
    mean_capture_latency_ms: float = 0.0
    mean_extraction_latency_ms: float = 0.0
    mean_normalization_latency_ms: float = 0.0
    mean_quality_latency_ms: float = 0.0
    mean_total_latency_ms: float = 0.0
    median_total_latency_ms: float = 0.0
    p95_total_latency_ms: float = 0.0
    max_total_latency_ms: float = 0.0
    valid_frame_ratio: float = 0.0


@dataclass
class PredictionResult:
    """
    Structured sign classification prediction emitted from a sliding temporal window.
    """
    class_id: int
    label: str
    gloss: str
    translation: str
    confidence: float
    probabilities: np.ndarray               # Full probability distribution [num_classes]
    logits: np.ndarray                      # Raw unnormalized model logits [num_classes]
    top_k: List[Tuple[int, str, float]]     # List of (class_id, label, probability)
    timestamp: float                        # Prediction timestamp
    window_start_time: float                # Timestamp of earliest frame in window
    window_end_time: float                  # Timestamp of latest frame in window
    window_frame_count: int                 # Number of frames in window (e.g. 45)
    inference_start_time: float             # When model forward pass started
    inference_end_time: float               # When model forward pass completed
    inference_latency_ms: float             # ST-GCN inference computation time
    input_quality: float                    # Mean landmark quality score across window
    is_valid_quality: bool                  # Whether window satisfies quality gate
    buffer_length: int                      # Current buffer length when inference ran
    device: str                             # Device used for inference ('cpu', 'cuda')

    def to_dict(self) -> Dict[str, Any]:
        """Serializes prediction result into dictionary representation."""
        return {
            "class_id": self.class_id,
            "label": self.label,
            "gloss": self.gloss,
            "translation": self.translation,
            "confidence": round(float(self.confidence), 4),
            "top_k": [(cid, lbl, round(float(p), 4)) for cid, lbl, p in self.top_k],
            "timestamp": round(self.timestamp, 4),
            "window_duration_s": round(self.window_end_time - self.window_start_time, 3),
            "inference_latency_ms": round(self.inference_latency_ms, 2),
            "input_quality": round(self.input_quality, 4),
            "is_valid_quality": self.is_valid_quality,
            "device": self.device
        }

    def format_console(self) -> str:
        """Formatted single-line string for console logging."""
        top_str = " | ".join([f"{lbl}: {p:.1%}" for _, lbl, p in self.top_k])
        return (
            f"Prediction: {self.label.upper()} ({self.confidence:.1%}) | "
            f"Latency: {self.inference_latency_ms:.1f}ms | Top-3: [{top_str}]"
        )

