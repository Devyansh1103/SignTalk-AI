"""
SignTalk AI - Real-Time Configuration Loader
Loads and validates configurations for camera capture, preprocessing,
landmark extraction, and quality control from YAML files or CLI arguments.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List


@dataclass
class CameraConfig:
    device_index: Union[int, str] = 0
    width: int = 640
    height: int = 480
    target_fps: float = 25.0
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 3
    reconnect_delay_sec: float = 1.0


@dataclass
class ProcessingConfig:
    flip_horizontal: bool = True
    color_format: str = "RGB"
    target_width: Optional[int] = None
    target_height: Optional[int] = None


@dataclass
class MediaPipeConfig:
    pose_model_path: str = "models/mediapipe/pose_landmarker_full.task"
    hand_model_path: str = "models/mediapipe/hand_landmarker.task"
    face_model_path: str = "models/mediapipe/face_landmarker.task"
    pose_detection_confidence: float = 0.50
    pose_presence_confidence: float = 0.50
    pose_tracking_confidence: float = 0.50
    hand_detection_confidence: float = 0.35
    hand_presence_confidence: float = 0.35
    hand_tracking_confidence: float = 0.35
    face_detection_confidence: float = 0.40
    face_presence_confidence: float = 0.40
    enable_hands: bool = True
    enable_pose: bool = True
    enable_face: bool = False


@dataclass
class NormalizationConfig:
    method: str = "torso_scale"
    center_point: str = "mid_shoulder"
    scale_reference: str = "shoulder_distance"
    min_scale_epsilon: float = 1.0e-4
    z_scale_factor: float = 1.0
    use_temporal_anchor_smoothing: bool = True
    anchor_smoothing_alpha: float = 0.25


@dataclass
class QualityConfig:
    min_valid_pose_ratio: float = 0.70
    min_valid_hand_ratio: float = 0.30
    overall_quality_threshold: float = 0.50
    max_consecutive_missing: int = 10
    min_valid_frames_in_sequence: int = 15
    jump_distance_threshold: float = 0.35
    weights: Dict[str, float] = field(default_factory=lambda: {
        "pose": 0.40,
        "dominant_hand": 0.40,
        "non_dominant_hand": 0.15,
        "face": 0.05
    })
    classification_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "good": 0.75,
        "acceptable": 0.50,
        "review": 0.35
    })


@dataclass
class StreamConfig:
    buffer_capacity: int = 120
    drop_policy: str = "drop_oldest"
    warmup_frames: int = 10


@dataclass
class RuntimeConfig:
    display_preview: bool = True
    preview_window_name: str = "SignTalk AI — Real-Time Landmark Stream"
    save_debug_frames: bool = False


@dataclass
class DebugConfig:
    record_landmarks: bool = False
    record_video: bool = False
    record_dir: str = "experiments/realtime_debug"


@dataclass
class TemporalBufferConfig:
    max_length: int = 45
    min_valid_frames: int = 15
    interpolate_missing_frames: bool = True
    max_consecutive_interpolated: int = 10


@dataclass
class SchedulerConfig:
    stride: int = 5
    drop_stale_windows: bool = True


@dataclass
class InferenceConfig:
    enabled: bool = True
    checkpoint_path: str = "experiments/stgcn/checkpoints/best_checkpoint.pt"
    device: str = "auto"
    top_k: int = 3
    warmup_iterations: int = 5
    record_predictions: bool = False
    predictions_csv: str = "results/realtime/predictions.csv"
    verbose_console: bool = False


@dataclass
class RealTimeConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    buffer: TemporalBufferConfig = field(default_factory=TemporalBufferConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str = "configs/realtime.yaml") -> "RealTimeConfig":
        """Loads configuration from YAML file with default fallback."""
        if not os.path.exists(yaml_path):
            return cls()

        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        cfg = cls()

        if "camera" in raw:
            c = raw["camera"]
            cfg.camera = CameraConfig(
                device_index=c.get("device_index", cfg.camera.device_index),
                width=int(c.get("width", cfg.camera.width)),
                height=int(c.get("height", cfg.camera.height)),
                target_fps=float(c.get("target_fps", cfg.camera.target_fps)),
                auto_reconnect=bool(c.get("auto_reconnect", cfg.camera.auto_reconnect)),
                max_reconnect_attempts=int(c.get("max_reconnect_attempts", cfg.camera.max_reconnect_attempts)),
                reconnect_delay_sec=float(c.get("reconnect_delay_sec", cfg.camera.reconnect_delay_sec))
            )

        if "processing" in raw:
            p = raw["processing"]
            tw = p.get("target_width")
            th = p.get("target_height")
            cfg.processing = ProcessingConfig(
                flip_horizontal=bool(p.get("flip_horizontal", cfg.processing.flip_horizontal)),
                color_format=str(p.get("color_format", cfg.processing.color_format)),
                target_width=int(tw) if tw is not None else None,
                target_height=int(th) if th is not None else None
            )

        if "mediapipe" in raw:
            m = raw["mediapipe"]
            cfg.mediapipe = MediaPipeConfig(
                pose_model_path=str(m.get("pose_model_path", cfg.mediapipe.pose_model_path)),
                hand_model_path=str(m.get("hand_model_path", cfg.mediapipe.hand_model_path)),
                face_model_path=str(m.get("face_model_path", cfg.mediapipe.face_model_path)),
                pose_detection_confidence=float(m.get("pose_detection_confidence", cfg.mediapipe.pose_detection_confidence)),
                pose_presence_confidence=float(m.get("pose_presence_confidence", cfg.mediapipe.pose_presence_confidence)),
                pose_tracking_confidence=float(m.get("pose_tracking_confidence", cfg.mediapipe.pose_tracking_confidence)),
                hand_detection_confidence=float(m.get("hand_detection_confidence", cfg.mediapipe.hand_detection_confidence)),
                hand_presence_confidence=float(m.get("hand_presence_confidence", cfg.mediapipe.hand_presence_confidence)),
                hand_tracking_confidence=float(m.get("hand_tracking_confidence", cfg.mediapipe.hand_tracking_confidence)),
                face_detection_confidence=float(m.get("face_detection_confidence", cfg.mediapipe.face_detection_confidence)),
                face_presence_confidence=float(m.get("face_presence_confidence", cfg.mediapipe.face_presence_confidence)),
                enable_hands=bool(m.get("enable_hands", cfg.mediapipe.enable_hands)),
                enable_pose=bool(m.get("enable_pose", cfg.mediapipe.enable_pose)),
                enable_face=bool(m.get("enable_face", cfg.mediapipe.enable_face))
            )

        if "normalization" in raw:
            n = raw["normalization"]
            cfg.normalization = NormalizationConfig(
                method=str(n.get("method", cfg.normalization.method)),
                center_point=str(n.get("center_point", cfg.normalization.center_point)),
                scale_reference=str(n.get("scale_reference", cfg.normalization.scale_reference)),
                min_scale_epsilon=float(n.get("min_scale_epsilon", cfg.normalization.min_scale_epsilon)),
                z_scale_factor=float(n.get("z_scale_factor", cfg.normalization.z_scale_factor)),
                use_temporal_anchor_smoothing=bool(n.get("use_temporal_anchor_smoothing", cfg.normalization.use_temporal_anchor_smoothing)),
                anchor_smoothing_alpha=float(n.get("anchor_smoothing_alpha", cfg.normalization.anchor_smoothing_alpha))
            )

        if "quality" in raw:
            q = raw["quality"]
            cfg.quality = QualityConfig(
                min_valid_pose_ratio=float(q.get("min_valid_pose_ratio", cfg.quality.min_valid_pose_ratio)),
                min_valid_hand_ratio=float(q.get("min_valid_hand_ratio", cfg.quality.min_valid_hand_ratio)),
                overall_quality_threshold=float(q.get("overall_quality_threshold", cfg.quality.overall_quality_threshold)),
                max_consecutive_missing=int(q.get("max_consecutive_missing", cfg.quality.max_consecutive_missing)),
                min_valid_frames_in_sequence=int(q.get("min_valid_frames_in_sequence", cfg.quality.min_valid_frames_in_sequence)),
                jump_distance_threshold=float(q.get("jump_distance_threshold", cfg.quality.jump_distance_threshold)),
                weights=q.get("weights", cfg.quality.weights),
                classification_thresholds=q.get("classification_thresholds", cfg.quality.classification_thresholds)
            )

        if "stream" in raw:
            s = raw["stream"]
            cfg.stream = StreamConfig(
                buffer_capacity=int(s.get("buffer_capacity", cfg.stream.buffer_capacity)),
                drop_policy=str(s.get("drop_policy", cfg.stream.drop_policy)),
                warmup_frames=int(s.get("warmup_frames", cfg.stream.warmup_frames))
            )

        if "runtime" in raw:
            r = raw["runtime"]
            cfg.runtime = RuntimeConfig(
                display_preview=bool(r.get("display_preview", cfg.runtime.display_preview)),
                preview_window_name=str(r.get("preview_window_name", cfg.runtime.preview_window_name)),
                save_debug_frames=bool(r.get("save_debug_frames", cfg.runtime.save_debug_frames))
            )

        if "debug" in raw:
            d = raw["debug"]
            cfg.debug = DebugConfig(
                record_landmarks=bool(d.get("record_landmarks", cfg.debug.record_landmarks)),
                record_video=bool(d.get("record_video", cfg.debug.record_video)),
                record_dir=str(d.get("record_dir", cfg.debug.record_dir))
            )

        if "buffer" in raw:
            b = raw["buffer"]
            cfg.buffer = TemporalBufferConfig(
                max_length=int(b.get("max_length", cfg.buffer.max_length)),
                min_valid_frames=int(b.get("min_valid_frames", cfg.buffer.min_valid_frames)),
                interpolate_missing_frames=bool(b.get("interpolate_missing_frames", cfg.buffer.interpolate_missing_frames)),
                max_consecutive_interpolated=int(b.get("max_consecutive_interpolated", cfg.buffer.max_consecutive_interpolated))
            )

        if "scheduler" in raw:
            sc = raw["scheduler"]
            cfg.scheduler = SchedulerConfig(
                stride=int(sc.get("stride", cfg.scheduler.stride)),
                drop_stale_windows=bool(sc.get("drop_stale_windows", cfg.scheduler.drop_stale_windows))
            )

        if "inference" in raw:
            inf = raw["inference"]
            cfg.inference = InferenceConfig(
                enabled=bool(inf.get("enabled", cfg.inference.enabled)),
                checkpoint_path=str(inf.get("checkpoint_path", cfg.inference.checkpoint_path)),
                device=str(inf.get("device", cfg.inference.device)),
                top_k=int(inf.get("top_k", cfg.inference.top_k)),
                warmup_iterations=int(inf.get("warmup_iterations", cfg.inference.warmup_iterations)),
                record_predictions=bool(inf.get("record_predictions", cfg.inference.record_predictions)),
                predictions_csv=str(inf.get("predictions_csv", cfg.inference.predictions_csv)),
                verbose_console=bool(inf.get("verbose_console", cfg.inference.verbose_console))
            )

        return cfg
