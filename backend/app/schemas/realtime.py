"""
SignTalk AI - Real-Time API Pydantic Schemas
Phase 5: WebSocket & REST Message Contracts.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = True
    device: str = "cpu"
    architecture: str = "ST-GCN + Linguistic Translation"
    vocabulary_size: int = 10
    version: str = "1.0.0"


class VocabularyItem(BaseModel):
    class_id: int
    label: str
    gloss: str
    english_translation: str
    hindi_translation: str


class VocabularyResponse(BaseModel):
    vocabulary_name: str = "mvp_10"
    num_classes: int = 10
    classes: List[VocabularyItem]


class ClientControlMessage(BaseModel):
    action: str = Field(..., description="Action: start, pause, resume, reset, stop")
    session_id: Optional[str] = None


class ClientFrameMessage(BaseModel):
    type: str = "frame"
    image_base64: Optional[str] = None
    timestamp: Optional[float] = None


class ServerPredictionPayload(BaseModel):
    class_id: Optional[int] = None
    label: Optional[str] = None
    gloss: Optional[str] = None
    confidence: float = 0.0
    input_quality: float = 0.0
    top_k: List[List[Any]] = Field(default_factory=list)


class ServerTranslationPayload(BaseModel):
    text: str = ""
    hindi_text: str = ""
    confidence: float = 0.0
    tokens: List[str] = Field(default_factory=list)
    source_signs: List[str] = Field(default_factory=list)
    is_final: bool = False


class ServerRealtimeMessage(BaseModel):
    type: str = "prediction"  # prediction, status, event, error
    session_id: str
    timestamp: float
    pipeline_state: str
    prediction: Optional[ServerPredictionPayload] = None
    smoothed_label: Optional[str] = None
    state_machine: str = "IDLE"
    sign_sequence: List[str] = Field(default_factory=list)
    translation: Optional[ServerTranslationPayload] = None
    stage_latencies_ms: Dict[str, float] = Field(default_factory=dict)
    fps: float = 0.0
    is_valid_input: bool = True
    message: Optional[str] = None
