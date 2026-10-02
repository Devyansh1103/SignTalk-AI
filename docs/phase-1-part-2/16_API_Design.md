# 16. API Design & Protocol Specifications: SignTalk AI

**Document ID:** STAI-P1P2-016  
**Project Name:** SignTalk AI  
**Document Status:** Approved Engineering Blueprint (Phase 1 — Part 2)  

---

## 1. RESTful API Contracts

All REST endpoints operate over HTTP/1.1 or HTTP/2, returning JSON payloads conforming to standard HTTP status codes.

### 1.1 Health Check Endpoint
* **Route:** `GET /healthz`
* **Response Status:** `200 OK` (Healthy) | `503 Service Unavailable` (Model not loaded)
* **Response Payload Schema:**
  ```json
  {
    "status": "healthy",
    "timestamp_utc": "2026-10-15T10:30:00Z",
    "version": "1.0.0-phase1",
    "model_loaded": true,
    "inference_device": "cuda:0",
    "active_sessions": 1
  }
  ```

### 1.2 Session Lifecycle Endpoint
* **Route:** `POST /api/v1/session`
* **Request Body:**
  ```json
  {
    "client_id": "client_web_01",
    "domain_mode": "healthcare",
    "target_language": "en"
  }
  ```
* **Response Status:** `201 Created`
* **Response Payload:**
  ```json
  {
    "session_id": "sess_8f3a9b1c",
    "created_at": "2026-10-15T10:30:15Z",
    "ws_stream_url": "/ws/translate/sess_8f3a9b1c",
    "buffer_capacity": 45,
    "inference_stride": 5
  }
  ```

### 1.3 Model Metadata Endpoint
* **Route:** `GET /api/v1/model/info`
* **Response Status:** `200 OK`
* **Response Payload:**
  ```json
  {
    "model_name": "STGCN_Transformer_Hybrid",
    "model_version": "stgcn-trans-v1.0.0",
    "vocabulary_size": 500,
    "mvp_classes": 50,
    "node_count": 93,
    "temporal_window": 45,
    "quantization": "FP32",
    "checksum_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  }
  ```

---

## 2. Real-Time WebSocket Streaming Protocol (`/ws/translate/{session_id}`)

### 2.1 Connection Lifecycle
1. **Handshake:** Client initiates WebSocket connection: `ws://localhost:8000/ws/translate/sess_8f3a9b1c`.
2. **Session Validation:** Server verifies that `session_id` exists in `SessionManager`. If invalid, closes with code `4404 Session Not Found`.
3. **Heartbeat:** Client transmits periodic `ping` messages every 15 seconds; server responds immediately with `pong`.

### 2.2 Client-to-Server Message Contracts

#### Message A: Stream Landmark Frame (`landmark_frame`)
```json
{
  "type": "landmark_frame",
  "frame_id": 1420,
  "timestamp_ms": 1729004523120,
  "landmarks": [
    [-0.142, 0.485, -0.021],
    [-0.155, 0.512, -0.019]
  ],
  "hand_detected": {
    "left": true,
    "right": true
  }
}
```

#### Message B: Session Control Directive (`session_control`)
```json
{
  "type": "session_control",
  "action": "pause"
}
```

---

### 2.3 Server-to-Client Message Contracts

#### Message A: Verified Translation Caption (`translation_caption`)
```json
{
  "type": "translation_caption",
  "frame_id": 1420,
  "timestamp_ms": 1729004523310,
  "status": "final",
  "text": "I have had a severe fever since yesterday.",
  "confidence": 0.884,
  "latency_metrics": {
    "stgcn_ms": 48.2,
    "transformer_ms": 78.5,
    "total_backend_ms": 132.1
  }
}
```

#### Message B: Diagnostic Warning & Repeat Prompt (`translation_warning`)
```json
{
  "type": "translation_warning",
  "warning_code": "LOW_CONFIDENCE",
  "message": "Confidence below threshold. Please repeat sign clearly.",
  "confidence": 0.382
}
```
