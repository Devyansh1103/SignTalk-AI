"""
SignTalk AI - Real-Time WebSocket Endpoint
Phase 5: Bidirectional Streaming for Frame Ingestion & Telemetry Distribution.
"""

import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.app.services.session_manager import SessionManager
from backend.app.schemas.realtime import ServerRealtimeMessage

logger = logging.getLogger("SignTalk.Backend.WebSocket")
router = APIRouter()


def get_session_manager():
    from backend.app.main import session_manager
    return session_manager


@router.websocket("/ws/realtime")
async def websocket_realtime_endpoint(websocket: WebSocket):
    """
    Bidirectional real-time WebSocket connection.
    Ingests video frames (binary JPEG/PNG or base64 JSON) and streams structured
    predictions, sign events, and linguistic translations to the web client.
    """
    await websocket.accept()
    sm = get_session_manager()
    session = sm.create_session()
    logger.info(f"WebSocket client connected. Assigned session: {session.session_id}")

    # Send initial welcome & connection confirmation
    init_msg = ServerRealtimeMessage(
        type="status",
        session_id=session.session_id,
        timestamp=0.0,
        pipeline_state=session.pipeline.state.value,
        state_machine="IDLE",
        message="Connected to SignTalk AI real-time inference engine."
    )
    await websocket.send_text(init_msg.model_dump_json())

    try:
        while True:
            # Receive either binary data or text JSON
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                # Binary frame ingestion (fastest path)
                try:
                    res = session.process_frame_bytes(message["bytes"])
                    await websocket.send_text(res.model_dump_json())
                except Exception as e:
                    logger.error(f"Error processing binary frame: {e}")
                    err_msg = ServerRealtimeMessage(
                        type="error",
                        session_id=session.session_id,
                        timestamp=0.0,
                        pipeline_state="ERROR",
                        message=f"Frame processing error: {str(e)}"
                    )
                    await websocket.send_text(err_msg.model_dump_json())

            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                except Exception:
                    continue

                action = data.get("action")
                msg_type = data.get("type")

                if action:
                    if action == "reset":
                        session.reset()
                        logger.info(f"Session {session.session_id} reset requested.")
                        status_msg = ServerRealtimeMessage(
                            type="status",
                            session_id=session.session_id,
                            timestamp=0.0,
                            pipeline_state=session.pipeline.state.value,
                            state_machine="IDLE",
                            message="Pipeline sequence and transcript reset."
                        )
                        await websocket.send_text(status_msg.model_dump_json())

                    elif action == "pause":
                        session.pause()
                        status_msg = ServerRealtimeMessage(
                            type="status",
                            session_id=session.session_id,
                            timestamp=0.0,
                            pipeline_state="PAUSED",
                            state_machine=session.pipeline.sign_state_machine.state.value,
                            message="Inference paused."
                        )
                        await websocket.send_text(status_msg.model_dump_json())

                    elif action == "resume":
                        session.resume()
                        status_msg = ServerRealtimeMessage(
                            type="status",
                            session_id=session.session_id,
                            timestamp=0.0,
                            pipeline_state="RUNNING",
                            state_machine=session.pipeline.sign_state_machine.state.value,
                            message="Inference resumed."
                        )
                        await websocket.send_text(status_msg.model_dump_json())

                    elif action == "stop":
                        session.pipeline.stop()
                        status_msg = ServerRealtimeMessage(
                            type="status",
                            session_id=session.session_id,
                            timestamp=0.0,
                            pipeline_state="STOPPED",
                            state_machine="IDLE",
                            message="Inference stopped."
                        )
                        await websocket.send_text(status_msg.model_dump_json())

                elif msg_type == "frame" or "image_base64" in data:
                    b64 = data.get("image_base64", "")
                    if b64:
                        try:
                            res = session.process_base64_frame(b64)
                            await websocket.send_text(res.model_dump_json())
                        except Exception as e:
                            logger.error(f"Error processing base64 frame: {e}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {session.session_id}")
    except Exception as e:
        logger.error(f"WebSocket exception in session {session.session_id}: {e}")
    finally:
        sm.close_session(session.session_id)
