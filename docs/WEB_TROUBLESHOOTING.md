# SignTalk AI: Web Application Troubleshooting & Operations Guide

**Project**: SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Phase**: Phase 5  

---

## 1. Camera Access & Media Devices

### Symptom: "Camera Access Denied" or Black Video Feed
- **Root Cause**: The browser has blocked webcam permissions for `http://localhost:5173`.
- **Resolution**:
  1. Click the lock or camera icon in your browser address bar.
  2. Toggle **Camera** permission to **Allow**.
  3. Reload the webpage.
  4. Ensure no other application (e.g., Zoom, Teams, OBS Studio) has exclusive locks on the physical webcam device.

### Symptom: `NotAllowedError` in Non-Localhost Environments
- **Root Cause**: Modern browsers (Chrome, Edge, Firefox, Safari) strictly require **Secure Contexts (HTTPS)** to allow `navigator.mediaDevices.getUserMedia()`.
- **Resolution**:
  - `localhost` and `127.0.0.1` are treated as secure contexts by default.
  - If accessing from a local network IP (e.g., `http://192.168.1.X:5173`), you must either configure an HTTPS reverse proxy (e.g., Caddy, NGINX) or enable `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.

---

## 2. WebSocket & Backend Connectivity

### Symptom: Badge Displays "Disconnected" or "Connection Error"
- **Root Cause**: The FastAPI backend server is not running or running on an unexpected port.
- **Verification**:
  1. Open a terminal and run:
     ```powershell
     curl http://localhost:8000/health
     ```
     Should return: `{"status":"healthy","version":"1.0.0",...}`.
  2. If connection fails, launch the backend server:
     ```powershell
     uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
     ```
  3. Ensure no firewall or antivirus is blocking TCP port `8000`.

### Symptom: Periodic WebSocket Disconnections
- **Root Cause**: Network timeouts or heavy GC pauses.
- **Handling**: The frontend `RealtimeClient` incorporates automatic exponential backoff reconnection up to 5 attempts. If connection does not restore, press `Reset` or refresh the tab.

---

## 3. Inference Latency & Performance Drops

### Symptom: Low FPS or Jerky Video (FPS < 10)
- **Root Cause**: Heavy CPU load during simultaneous MediaPipe holistic extraction and PyTorch ST-GCN graph convolutions.
- **Resolution**:
  1. Toggle open the **Developer HUD** (`S` key) to inspect `stage_latencies_ms`.
  2. If `mediapipe` latency exceeds 50 ms, lower camera resolution or frame capture rate in `frontend/src/hooks/useRealtime.ts` (default is 15 FPS).
  3. Check system power profile: Ensure laptop is connected to AC power and not in battery saver mode.
  4. Ensure PyTorch is using multiple CPU threads (`torch.set_num_threads(4)`).

---

## 4. MediaPipe Landmark Tracking Issues

### Symptom: "Hands Not Detected" or Persistent "UNCERTAIN" Status
- **Root Cause**: Poor lighting contrast or hands positioned outside camera boundary.
- **Resolution**:
  1. Ensure ambient lighting is in front of the signer, avoiding harsh backlighting (e.g., bright window behind the signer).
  2. Wear clothing that contrasts with your skin tone (e.g., dark shirt for lighter skin, light shirt for darker skin).
  3. Follow the framing silhouette in [CameraGuide.tsx](file:///d:/SignAI/frontend/src/components/camera/CameraGuide.tsx) to ensure both hands, wrists, and shoulders are in view.
  4. Keep hands at chest or chin height when initiating a sign gesture.
