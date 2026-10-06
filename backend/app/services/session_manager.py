"""
SignTalk AI - WebSocket Session Manager
Phase 5: Concurrent Session Lifecycle Management & Resource Cleanup.
"""

import os
import sys
import uuid
import logging
from typing import Dict, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.realtime.model_runner import STGCNRunner
from src.realtime.realtime_config import RealTimeConfig
from backend.app.services.inference_service import SessionInferenceService

logger = logging.getLogger("SignTalk.Backend.SessionManager")


class SessionManager:
    """
    Manages active client sessions, shares pre-warmed model weights across sessions,
    and guarantees resource reclamation upon client disconnection.
    """

    def __init__(self, config: Optional[RealTimeConfig] = None):
        self.config = config or RealTimeConfig()
        self._sessions: Dict[str, SessionInferenceService] = {}
        self._shared_model_runner: Optional[STGCNRunner] = None
        self._initialize_shared_model()

    def _initialize_shared_model(self) -> None:
        """Pre-loads ST-GCN weights once globally."""
        chk = self.config.inference.checkpoint_path
        if os.path.exists(chk):
            logger.info(f"Pre-warming shared ST-GCN model from {chk}...")
            self._shared_model_runner = STGCNRunner(
                checkpoint_path=chk,
                device=self.config.inference.device,
                top_k=self.config.inference.top_k,
                warmup_iterations=self.config.inference.warmup_iterations
            )
        else:
            logger.warning(f"Checkpoint not found at {chk}. Model will load on demand if created.")

    def create_session(self, session_id: Optional[str] = None) -> SessionInferenceService:
        """Instantiates an isolated session with its own rolling buffers and state machine."""
        sid = session_id or f"sess_{uuid.uuid4().hex[:10]}"
        if sid in self._sessions:
            logger.info(f"Reusing existing session: {sid}")
            return self._sessions[sid]

        logger.info(f"Creating isolated inference session: {sid}")
        service = SessionInferenceService(
            session_id=sid,
            config=self.config,
            shared_model_runner=self._shared_model_runner
        )
        self._sessions[sid] = service
        return service

    def get_session(self, session_id: str) -> Optional[SessionInferenceService]:
        """Retrieves an active session by ID."""
        return self._sessions.get(session_id)

    def close_session(self, session_id: str) -> None:
        """Cleans up and terminates an active session."""
        service = self._sessions.pop(session_id, None)
        if service is not None:
            logger.info(f"Closing and reclaiming session: {session_id}")
            service.close()

    def active_session_count(self) -> int:
        return len(self._sessions)
