# SignTalk AI — Real-Time Pipeline Contract Specification

**Project:** SignTalk AI: Real-Time Indian Sign Language Translation Platform  
**Document:** End-to-End Real-Time Pipeline Data Contract  
**Milestone:** Phase 4 — Part 4  
**Date:** October 6, 2026  

---

## 1. Overview & Data Flow Topology

The SignTalk AI real-time engine operates as a strictly typed, unidirectional dataflow pipeline from physical optical acquisition to linguistic caption display:

```text
FramePacket (Camera)
      │
      ▼
LandmarkFrame (MediaPipe Extraction + Normalization + Quality)
      │
      ▼
TemporalBuffer (FIFO Window Queue: T=45)
      │
      ▼
Input Tensor (PyTorch Float32 [1, 3, 45, 93])
      │
      ▼
PredictionResult (ST-GCN Logits & Softmax Probabilities)
      │
      ▼
FilteredPrediction (Confidence & Quality Gate)
      │
      ▼
SmoothedPrediction (Temporal Majority Voting Consensus)
      │
      ▼
SignState (Finite State Machine Transition)
      │
      ▼
SignEvent (Debounced & Finalized Sign Gesture)
      │
      ▼
SignSequence (Chronological Sign Buffer)
      │
      ▼
TranslationResult (Natural Language Sentence / Phrase)
      │
      ▼
LiveTranscript (Structured Live Caption & History State)
```

---

## 2. Granular Stage-by-Stage Object Contracts

### Stage 1: Raw Frame Packet
* **Object Name:** `FramePacket` ([`src/realtime/types.py`](file:///d:/SignAI/src/realtime/types.py))
* **Producer:** `Camera` ([`src/realtime/camera.py`](file:///d:/SignAI/src/realtime/camera.py))
* **Consumer:** `RealTimeLandmarkStream`
* **Fields & Types:**
  * `frame_id`: `int` (Monotonically increasing sequence identifier $\ge 0$)
  * `timestamp`: `float` (Monotonic wall-clock timestamp in seconds)
  * `capture_time`: `float` (Duration of frame read in seconds)
  * `image`: `np.ndarray` (BGR image, shape `(H, W, 3)`, `dtype=uint8`)
  * `width`: `int` (Horizontal resolution, typically 640)
  * `height`: `int` (Vertical resolution, typically 480)
  * `is_rgb`: `bool` (False for OpenCV BGR; True when converted)
  * `is_flipped`: `bool` (True if horizontally mirrored for selfie preview)
* **Lifecycle & Ownership:** Ephemeral; passed by reference. Image memory is overwritten or collected once landmark extraction completes.
* **Error Behavior:** If camera read fails, camera raises `CameraReadError` or triggers automatic reconnection.

---

### Stage 2: Validated Landmark Frame
* **Object Name:** `LandmarkFrame` ([`src/realtime/types.py`](file:///d:/SignAI/src/realtime/types.py))
* **Producer:** `RealTimeLandmarkStream.process_frame()`
* **Consumer:** `TemporalBuffer`
* **Fields & Types:**
  * `frame_id`: `int` (Inherited from `FramePacket`)
  * `timestamp`: `float` (Inherited from `FramePacket`)
  * `raw_coords`: `np.ndarray` (Normalized image space $[0, 1]$, shape `(93, 3)`, `float32`)
  * `normalized_coords`: `np.ndarray` (Torso-centered, shoulder-scale normalized, shape `(93, 3)`, `float32`)
  * `mask`: `np.ndarray` (Boolean visibility mask, shape `(93,)`, `bool`)
  * `visibility`: `np.ndarray` (Confidence score per node, shape `(93,)`, `float32`)
  * `modality_stats`: `ModalityDetection` (Presence flags for left hand, right hand, pose, face)
  * `quality`: `QualityReport` (`is_valid: bool`, `quality_score: float` $[0.0, 1.0]$, `status: str`)
* **Invariants:** `normalized_coords.shape == (93, 3)`, no NaNs, no Infs.
* **Error Behavior:** If tracking is completely lost, `is_valid` is set to `False`, quality score is $0.0$, and coordinates default to zeros.

---

### Stage 3: Temporal Rolling Window
* **Object Name:** `List[LandmarkFrame]` ($T=45$)
* **Producer:** `TemporalBuffer.get_window()` ([`src/realtime/temporal_buffer.py`](file:///d:/SignAI/src/realtime/temporal_buffer.py))
* **Consumer:** `STGCNRunner.window_to_tensor()`
* **Fields & Types:** Fixed-length list containing exactly $45$ chronologically ordered `LandmarkFrame` instances.
* **Duration:** $45 \text{ frames} \times 40\text{ ms} = 1.80\text{ seconds}$ of physical gesture observation.
* **Error & Interpolation Policy:** If isolated frames are dropped ($\le 10$ frames), linear interpolation synthesizes intermediate coordinates. If $> 10$ consecutive frames are missing, the window is marked invalid.

---

### Stage 4: Model Input Tensors
* **Object Name:** `Tuple[torch.Tensor, torch.Tensor]`
* **Producer:** `STGCNRunner.window_to_tensor()`
* **Consumer:** `SignSTGCN.forward()`
* **Dimensions & Dtypes:**
  * Landmark Tensor $X$: `torch.FloatTensor` of shape `[1, 3, 45, 93]`
    * Dimension 0 ($B$): Batch size, strictly $1$
    * Dimension 1 ($C$): Channels $(x, y, z)$, strictly $3$
    * Dimension 2 ($T$): Temporal sequence length, strictly $45$
    * Dimension 3 ($V$): Multimodal graph nodes, strictly $93$
  * Mask Tensor $M$: `torch.FloatTensor` of shape `[1, 1, 45, 93]`
* **Device:** Target execution device (`torch.device("cpu")` or `torch.device("cuda")`).
* **Validation Assertions:** Raises `ShapeValidationError` if dimensions, channels, or types deviate.

---

### Stage 5: Raw Prediction Result
* **Object Name:** `PredictionResult` ([`src/realtime/types.py`](file:///d:/SignAI/src/realtime/types.py))
* **Producer:** `STGCNRunner.predict_window()`
* **Consumer:** `ConfidenceFilter`, `PredictionHistory`
* **Fields & Types:**
  * `class_id`: `int` ($0 \le \text{class\_id} \le 9$)
  * `label`: `str` (Canonical label, e.g. `"hello"`)
  * `gloss`: `str` (Linguistic gloss, e.g. `"HELLO"`)
  * `translation`: `str` (Default translation, e.g. `"Hello"`)
  * `confidence`: `float` (Softmax probability $[0.0, 1.0]$)
  * `probabilities`: `np.ndarray` (Shape `(10,)`, `float32`)
  * `logits`: `np.ndarray` (Shape `(10,)`, `float32`)
  * `top_k`: `List[Tuple[int, str, float]]` (Top-3 ranked candidates)
  * `timestamp`: `float` (Wall-clock prediction timestamp)
  * `inference_latency_ms`: `float` (Forward pass time in milliseconds)
  * `input_quality`: `float` (Window mean quality score $[0.0, 1.0]$)
  * `is_valid_quality`: `bool` (True if window has $\ge 15$ valid frames)
  * `device`: `str` (`"cpu"` or `"cuda"`)

---

### Stage 6: Filtered & Smoothed Predictions
* **Object Name:** `FilteredPrediction` & `SmoothedPrediction`
* **Producers:** `ConfidenceFilter.filter_prediction()`, `BaseSmoother.smooth()`
* **Fields & Types:**
  * Model confidence score evaluated against $\tau = 0.65$.
  * If $\text{confidence} < 0.65$ or $\text{input\_quality} < 0.40$:
    * Tagged `UNCERTAIN` (`class_id = -1`).
  * If majority voting over recent $N=5$ predictions fails to achieve quorum $K=3$:
    * Tagged `UNSTABLE` (`class_id = -1`).

---

### Stage 7: Sign Event
* **Object Name:** `SignEvent` ([`src/realtime/sign_event.py`](file:///d:/SignAI/src/realtime/sign_event.py))
* **Producer:** `EventDeduplicator.process()` after `SignStateMachine` transition
* **Consumer:** `SignSequenceBuffer`
* **Fields & Types:**
  * `event_id`: `str` (Unique identifier, e.g. `"evt_7f1a8b92"`)
  * `class_id`: `int` (Canonical class index $0 \dots 9$)
  * `label`: `str` (Lower-case gloss label, e.g. `"hello"`)
  * `gloss`: `str` (Upper-case gloss label, e.g. `"HELLO"`)
  * `start_time`: `float` (Monotonic onset time in seconds)
  * `end_time`: `float` (Monotonic conclusion time in seconds)
  * `duration_ms`: `float` (Elapsed time in milliseconds)
  * `confidence`: `float` (Mean confidence during active state $[0.0, 1.0]$)
  * `input_quality`: `float` (Average landmark tracking quality $[0.0, 1.0]$)
  * `stability_score`: `float` (Consensus ratio $[0.0, 1.0]$)

---

### Stage 8: Sign Sequence Buffer
* **Object Name:** `SignSequenceBuffer` ([`src/realtime/sign_sequence.py`](file:///d:/SignAI/src/realtime/sign_sequence.py))
* **Consumer:** `RealTimeTranslator`
* **Operations:**
  * `get_sequence() -> List[SignEvent]`: Chronological event history.
  * `get_labels() -> List[str]`: Sequence of string glosses.
  * `format_gloss_string() -> str`: Concatenated sequence (e.g. `"HELLO -> GOOD -> MONDAY"`).
  * `clear()`: Purges buffer.

---

### Stage 9: Translation Result
* **Object Name:** `TranslationResult` ([`src/realtime/translator.py`](file:///d:/SignAI/src/realtime/translator.py))
* **Producer:** `RealTimeTranslator.translate()`
* **Consumer:** `LiveTranscript`
* **Fields & Types:**
  * `text`: `str` (Natural English translation, e.g. `"Hello, good Monday!"`)
  * `hindi_text`: `str` (Natural Hindi translation, e.g. `"नमस्ते, अच्छा सोमवार!"`)
  * `confidence`: `float` (Linguistic confidence score $[0.0, 1.0]$)
  * `tokens`: `List[str]` (Decoded lexical and grammatical tokens)
  * `source_signs`: `List[str]` (Input sign glosses consumed)
  * `timestamp`: `float` (Translation timestamp in seconds)
  * `status`: `str` (`"SUCCESS"`, `"LOW_CONFIDENCE"`, `"INSUFFICIENT_INPUT"`, `"NO_TRANSLATION"`, `"ERROR"`)
  * `is_final`: `bool` (True if committed upon pause/timeout; False if streaming preview)

---

### Stage 10: Live Transcript State
* **Object Name:** `LiveTranscriptState`
* **Producer:** `RealtimePipeline.get_transcript()`
* **Consumer:** Developer Overlay & Phase 5 Frontend Engine
* **Fields & Types:**
  * `current_sign`: `Optional[str]` (Active ongoing sign label)
  * `current_sign_confidence`: `float`
  * `current_phrase_glosses`: `List[str]` (Accumulated active glosses)
  * `current_phrase_text`: `str` (Provisional translation text)
  * `finalized_phrases`: `List[Dict[str, Any]]` (Chronological record of committed sentences with timestamps)
  * `pipeline_status`: `str` (`"RUNNING"`, `"PAUSED"`, `"STOPPED"`, `"ERROR"`)
