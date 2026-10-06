"""
SignTalk AI - Real-Time Inference Service
Phase 5: Session-Isolated Pipeline Adapter for Web Clients.
"""

import os
import sys
import time
import base64
import logging
from typing import Optional, Dict, Any
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.realtime.realtime_config import RealTimeConfig
from src.realtime.realtime_pipeline import RealtimePipeline, PipelineState
from src.realtime.model_runner import STGCNRunner
from backend.app.schemas.realtime import (
    ServerRealtimeMessage,
    ServerPredictionPayload,
    ServerTranslationPayload
)

logger = logging.getLogger("SignTalk.Backend.InferenceService")


class SessionInferenceService:
    """
    Session-isolated inference service coordinating a dedicated RealtimePipeline instance.
    Decodes web client video frames, manages session state, and serializes AI telemetry.
    """

    def __init__(
        self,
        session_id: str,
        config: Optional[RealTimeConfig] = None,
        shared_model_runner: Optional[STGCNRunner] = None,
    ):
        self.session_id = session_id
        self.config = config or RealTimeConfig()

        # Shared model weights can be reused across sessions to save memory,
        # but buffers, state machines, and sequence transcripts remain strictly isolated!
        self.pipeline = RealtimePipeline(
            config=self.config,
            camera=None,  # Frames arrive via WebSocket
            model_runner=shared_model_runner
        )
        self.pipeline.initialize()
        self._frame_count = 0
        self._is_active = True

    def process_frame_bytes(self, image_bytes: bytes) -> ServerRealtimeMessage:
        """Decodes raw JPEG/PNG image bytes and executes a pipeline step."""
        np_arr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode image frame.")
        return self.process_bgr_frame(bgr)

    def process_base64_frame(self, b64_str: str) -> ServerRealtimeMessage:
        """Decodes base64 data-URI or raw base64 string to BGR and steps pipeline."""
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        raw_bytes = base64.b64decode(b64_str)
        return self.process_frame_bytes(raw_bytes)

    def process_bgr_frame(self, bgr_frame: np.ndarray) -> ServerRealtimeMessage:
        """Processes a single BGR frame through the session pipeline."""
        self._frame_count += 1
        output = self.pipeline.step(bgr_frame)

        # Build response payload
        pred_payload = None
        if output.prediction is not None:
            pred_payload = ServerPredictionPayload(
                class_id=output.prediction.class_id,
                label=output.prediction.label,
                gloss=output.prediction.gloss,
                confidence=round(output.prediction.confidence, 4),
                input_quality=round(output.prediction.input_quality, 4),
                top_k=[[idx, lbl, round(conf, 4)] for idx, lbl, conf in output.prediction.top_k]
            )

        trans_payload = None
        if output.translation is not None:
            trans_payload = ServerTranslationPayload(
                text=output.translation.text,
                hindi_text=output.translation.hindi_text,
                confidence=round(output.translation.confidence, 4),
                tokens=list(output.translation.tokens),
                source_signs=list(output.translation.source_signs),
                is_final=output.translation.is_final
            )

        return ServerRealtimeMessage(
            type="prediction",
            session_id=self.session_id,
            timestamp=output.timestamp,
            pipeline_state=output.pipeline_state,
            prediction=pred_payload,
            smoothed_label=output.smoothed.label if output.smoothed else None,
            state_machine=output.state,
            sign_sequence=list(output.sign_sequence),
            translation=trans_payload,
            stage_latencies_ms={k: round(v, 2) for k, v in output.stage_latencies_ms.items()},
            fps=round(output.fps, 1),
            is_valid_input=output.is_valid_input
        )

    def pause(self) -> None:
        self.pipeline.pause()

    def resume(self) -> None:
        self.pipeline.resume()

    def reset(self) -> None:
        self.pipeline.reset()

    def close(self) -> None:
        self._is_active = False
        self.pipeline.close()
