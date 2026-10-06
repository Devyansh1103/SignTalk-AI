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

from src.realtime.prediction_history import PredictionRecord, PredictionHistory
from src.realtime.confidence_filter import FilteredPrediction, ConfidenceFilter
from src.realtime.smoothing import (
    BaseSmoother,
    SmoothedPrediction,
    MajorityVoteSmoother,
    ConfidenceWeightedSmoother,
    TemporalStabilitySmoother,
)
from src.realtime.sign_event import SignEvent
from src.realtime.sign_state_machine import SignState, SignStateMachine
from src.realtime.event_deduplicator import EventDeduplicator
from src.realtime.sign_sequence import SignSequenceBuffer
from src.realtime.temporal_metrics import (
    compute_prediction_flip_rate,
    compute_stability_durations,
    compute_duplicate_rate,
    compute_event_detection_latency,
    evaluate_temporal_pipeline,
)
from src.realtime.performance import PerformanceProfiler, StageTimer
from src.realtime.translator import RealTimeTranslator, TranslationResult
from src.realtime.realtime_pipeline import RealtimePipeline, PipelineOutput, PipelineState

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
    "RealTimeSignPredictor",
    # Confidence & History
    "PredictionRecord",
    "PredictionHistory",
    "FilteredPrediction",
    "ConfidenceFilter",
    # Smoothing
    "BaseSmoother",
    "SmoothedPrediction",
    "MajorityVoteSmoother",
    "ConfidenceWeightedSmoother",
    "TemporalStabilitySmoother",
    # Events & Sequence
    "SignEvent",
    "SignState",
    "SignStateMachine",
    "EventDeduplicator",
    "SignSequenceBuffer",
    # Metrics
    "compute_prediction_flip_rate",
    "compute_stability_durations",
    "compute_duplicate_rate",
    "compute_event_detection_latency",
    "evaluate_temporal_pipeline",
    # Phase 4 Part 4 - End-to-End Orchestration, Translation & Profiling
    "PerformanceProfiler",
    "StageTimer",
    "RealTimeTranslator",
    "TranslationResult",
    "RealtimePipeline",
    "PipelineOutput",
    "PipelineState",
]

