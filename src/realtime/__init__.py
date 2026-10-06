"""
SignTalk AI - Real-Time Streaming and Inference Architecture
Phase 4 Part 1 & Part 2: Real-Time Landmark Stream & ST-GCN Temporal Prediction.
"""

from src.realtime.types import (
    FramePacket,
    ModalityDetection,
    QualityReport,
    LandmarkFrame,
    StreamMetrics,
    PredictionResult
)
from src.realtime.realtime_config import (
    RealTimeConfig,
    CameraConfig,
    ProcessingConfig,
    MediaPipeConfig,
    NormalizationConfig,
    QualityConfig,
    StreamConfig,
    RuntimeConfig,
    DebugConfig,
    TemporalBufferConfig,
    SchedulerConfig,
    InferenceConfig
)
from src.realtime.camera import (
    Camera,
    CameraError,
    CameraInitializationError,
    CameraReadError
)
from src.realtime.frame_processor import FrameProcessor
from src.realtime.landmark_stream import RealTimeLandmarkStream
from src.realtime.temporal_buffer import TemporalBuffer
from src.realtime.inference_scheduler import InferenceScheduler
from src.realtime.model_runner import (
    STGCNRunner,
    ModelRunnerError,
    ShapeValidationError
)
from src.realtime.prediction_sink import (
    BasePredictionSink,
    ConsolePredictionSink,
    PredictionLoggerSink,
    CallbackPredictionSink,
    CompositePredictionSink
)
from src.realtime.realtime_predictor import RealTimeSignPredictor

__all__ = [
    # Types
    "FramePacket",
    "ModalityDetection",
    "QualityReport",
    "LandmarkFrame",
    "StreamMetrics",
    "PredictionResult",
    # Config
    "RealTimeConfig",
    "CameraConfig",
    "ProcessingConfig",
    "MediaPipeConfig",
    "NormalizationConfig",
    "QualityConfig",
    "StreamConfig",
    "RuntimeConfig",
    "DebugConfig",
    "TemporalBufferConfig",
    "SchedulerConfig",
    "InferenceConfig",
    # Ingestion & Streaming
    "Camera",
    "CameraError",
    "CameraInitializationError",
    "CameraReadError",
    "FrameProcessor",
    "RealTimeLandmarkStream",
    # Temporal & Inference
    "TemporalBuffer",
    "InferenceScheduler",
    "STGCNRunner",
    "ModelRunnerError",
    "ShapeValidationError",
    # Prediction Sinks
    "BasePredictionSink",
    "ConsolePredictionSink",
    "PredictionLoggerSink",
    "CallbackPredictionSink",
    "CompositePredictionSink",
    # Pipeline Coordinator
    "RealTimeSignPredictor"
]
