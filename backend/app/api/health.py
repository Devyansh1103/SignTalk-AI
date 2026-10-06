"""
SignTalk AI - Health & Metadata REST Endpoints
Phase 5: Status, Hardware Observability & Supported Vocabulary Manifest.
"""

from fastapi import APIRouter, Depends
import platform
import torch

from src.data.label_map import CANONICAL_CLASSES, NUM_CLASSES
from backend.app.schemas.realtime import (
    HealthResponse,
    VocabularyResponse,
    VocabularyItem
)
from backend.app.services.session_manager import SessionManager

router = APIRouter()


# Dependency injection helper
def get_session_manager():
    from backend.app.main import session_manager
    return session_manager


@router.get("/health", response_model=HealthResponse)
def health_check(sm: SessionManager = Depends(get_session_manager)):
    """Liveness probe verifying server health and model readiness."""
    has_model = sm._shared_model_runner is not None
    device_name = str(sm.config.inference.device)
    return HealthResponse(
        status="ok",
        model_loaded=has_model,
        device=device_name,
        architecture="ST-GCN + Multilingual Linguistic Translation",
        vocabulary_size=NUM_CLASSES,
        version="1.0.0"
    )


@router.get("/api/status")
def system_status(sm: SessionManager = Depends(get_session_manager)):
    """System status including active sessions, CPU/GPU telemetry, and configuration."""
    return {
        "status": "online",
        "active_sessions": sm.active_session_count(),
        "platform": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": sm.config.inference.device,
        "supported_vocabulary": NUM_CLASSES,
    }


@router.get("/api/vocabulary", response_model=VocabularyResponse)
def get_vocabulary():
    """Returns the certified 10-class Indian Sign Language vocabulary with translations."""
    from src.realtime.translator import HINDI_LEXICON

    items = []
    for cid, info in sorted(CANONICAL_CLASSES.items()):
        items.append(VocabularyItem(
            class_id=cid,
            label=info.label,
            gloss=info.gloss,
            english_translation=info.translation,
            hindi_translation=HINDI_LEXICON.get(info.label, "")
        ))

    return VocabularyResponse(
        vocabulary_name="mvp_10",
        num_classes=NUM_CLASSES,
        classes=items
    )
