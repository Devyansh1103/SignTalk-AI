"""
SignTalk AI - FastAPI Application Server
Phase 5: Real-Time Web Application Backend & WebSocket Server.

Run with:
  uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.app.services.session_manager import SessionManager
from backend.app.api.health import router as health_router
from backend.app.api.realtime import router as realtime_router

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("SignTalk.Backend")

# Global session manager instance
session_manager = SessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SignTalk AI backend service starting up...")
    yield
    logger.info("SignTalk AI backend shutting down, cleaning up all active sessions...")
    for sid in list(session_manager._sessions.keys()):
        session_manager.close_session(sid)


app = FastAPI(
    title="SignTalk AI — Real-Time Inference Platform",
    description="Real-Time Indian Sign Language Recognition & Translation WebSocket API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware for local frontend development and production hosting
allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(health_router, tags=["Health & Status"])
app.include_router(realtime_router, tags=["Real-Time Streaming"])


@app.get("/")
def root():
    return {
        "service": "SignTalk AI Real-Time Translation API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "websocket_endpoint": "/ws/realtime"
    }
