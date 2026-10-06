# SignTalk AI: Frontend-Backend Interface Contract

**Project**: SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Specification Version**: 1.0.0  
**Status**: Certified & Implemented  

---

## 1. Overview & Transport Protocol

SignTalk AI relies on two primary communication mechanisms:
1. **REST API**: Stateless discovery, health checking, and vocabulary metadata.
2. **WebSocket API**: Low-latency, full-duplex streaming of camera frames, control signals, and inference telemetry.

Default base URLs:
- REST: `http://localhost:8000`
- WebSocket: `ws://localhost:8000/ws/realtime`

---

## 2. REST Endpoints

### 2.1 Health Check
- **Path**: `GET /health`
- **Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": 1740000000.0,
  "service": "SignTalk-AI-Realtime-Backend"
}
```

### 2.2 System & Model Status
- **Path**: `GET /api/status`
- **Response**:
```json
{
  "status": "ready",
  "active_sessions": 1,
  "supported_vocabulary_count": 10,
  "model_checkpoint": "experiments/stgcn/checkpoints/best_checkpoint.pt",
  "device": "cpu",
  "target_fps": 15.0
}
```

### 2.3 Vocabulary Catalog
- **Path**: `GET /api/vocabulary`
- **Response**:
```json
{
  "count": 10,
  "vocabulary": [
    {
      "id": 0,
      "gloss": "hello",
      "display_name": "Hello",
      "hindi_translation": "नमस्ते",
      "category": "greetings",
      "description": "Open hand raised near temple moving outward."
    },
    {
      "id": 1,
      "gloss": "thankyou",
      "display_name": "Thank You",
      "hindi_translation": "धन्यवाद",
      "category": "politeness",
      "description": "Fingertips touching chin moving forward and down."
    }
  ]
}
```

---

## 3. WebSocket API (`/ws/realtime`)

The WebSocket endpoint provides real-time bidirectional messaging. The client establishes a single persistent connection per user session.

### 3.1 Client-to-Server Messages

#### A. Video Frame Stream (`frame`)
The client captures canvas frames at a throttled rate (target 15 FPS) and sends them as JPEG base64 payloads:
```json
{
  "type": "frame",
  "data": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "timestamp": 1740000001.123
}
```

#### B. Pipeline Control Commands (`control`)
The client can control the server-side pipeline lifecycle:
```json
{
  "type": "control",
  "action": "start" | "pause" | "resume" | "reset" | "stop"
}
```
- `start`: Transitions pipeline from initialized/idle to running.
- `pause`: Temporarily halts inference without clearing the temporal buffer.
- `resume`: Continues inference from paused state.
- `reset`: Flushes temporal buffer, state machine, and pending phrase translations.
- `stop`: Gracefully closes pipeline session.

---

### 3.2 Server-to-Client Messages

The server streams real-time updates corresponding to frame observations:

#### A. Prediction & Telemetry Stream (`inference_result`)
```json
{
  "type": "inference_result",
  "timestamp": 1740000001.205,
  "frame_idx": 42,
  "pipeline_state": "RUNNING",
  "landmarks_detected": true,
  "landmarks": {
    "left_hand": [[0.52, 0.45, -0.01], ...],
    "right_hand": [[0.60, 0.48, -0.02], ...],
    "pose": [[0.55, 0.30, 0.0], ...]
  },
  "prediction": {
    "sign": "hello",
    "confidence": 0.884,
    "is_stable": true,
    "top_predictions": [
      {"sign": "hello", "confidence": 0.884},
      {"sign": "thankyou", "confidence": 0.071},
      {"sign": "good", "confidence": 0.025}
    ]
  },
  "smoothed_label": "hello",
  "state_machine": "SIGNING",
  "sign_sequence": ["hello"],
  "translation": {
    "text": "Hello",
    "hindi_text": "नमस्ते",
    "is_finalized": false
  },
  "stage_latencies_ms": {
    "mediapipe": 28.4,
    "buffer_update": 0.2,
    "sliding_window": 0.1,
    "stgcn_inference": 94.2,
    "smoothing": 0.3,
    "translation": 0.1,
    "end_to_end": 123.3
  }
}
```

#### B. System Status Notifications (`status`)
```json
{
  "type": "status",
  "status": "ready" | "running" | "paused" | "reset_complete" | "stopped",
  "message": "Pipeline reset complete."
}
```

#### C. Error Notifications (`error`)
```json
{
  "type": "error",
  "error": "Failed to decode base64 video frame",
  "code": "FRAME_DECODE_ERROR"
}
```

---

## 4. Lifecycle & Reconnection Strategy

1. **Mounting**: When the web application loads, it initializes `RealtimeClient`.
2. **Exponential Backoff**: If connection drops unexpectedly, the client retries up to 5 times with backoff intervals (`1s`, `2s`, `4s`, `8s`, `16s`).
3. **Session Tear-down**: When the user leaves or pauses the tab, an explicit `control: stop` message is transmitted, allowing the server to release MediaPipe resources and session memory cleanly.
