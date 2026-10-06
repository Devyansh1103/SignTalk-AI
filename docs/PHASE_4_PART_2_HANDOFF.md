# SignTalk AI — Phase 4 Part 2 Handoff Package

**Document ID:** `DOC-P4P2-HANDOFF-001`  
**From:** Phase 4 — Part 1 (Real-Time Camera & Landmark Streaming)  
**To:** Phase 4 — Part 2 (Temporal Sequence Buffer & Real-Time ST-GCN Inference)  
**Date:** October 2026  
**Status:** READY FOR TEMPORAL BUFFER CONSUMPTION  

---

## 1. Executive Summary

Phase 4 Part 1 has established a production-grade, tested real-time ingestion layer:
$$\text{Camera} \to \text{Frame Capture} \to \text{Vision Preprocessor} \to \text{MediaPipe} \to \text{Fusion} \to \text{Torso-Scale Normalizer} \to \text{Quality Gate} \to \text{LandmarkStream}$$

The stream continuously produces individual `LandmarkFrame` instances validated against the canonical 93-node multimodal schema.

Phase 4 Part 2 will consume this stream, accumulate incoming frames into a sliding temporal ring buffer ($T=45$ frames @ 25 FPS), apply linear interpolation across dropped frames, and execute real-time inference using the selected checkpoint (`experiments/stgcn/checkpoints/best_checkpoint.pt`).

---

## 2. Ingestion Interface Contract for Part 2

```text
Stream Object:           src.realtime.RealTimeLandmarkStream
Frame Object:            src.realtime.types.LandmarkFrame
Coordinates Format:      frame.normalized_coords -> numpy.ndarray of shape (93, 3)
Detection Mask:          frame.mask -> numpy.ndarray of shape (93,) dtype bool
Visibility Scores:       frame.visibility -> numpy.ndarray of shape (93,) dtype float32
PyTorch Frame Tensor:    frame.to_stgcn_frame_tensor() -> torch.FloatTensor [3, 1, 93]
Quality Gate:            frame.quality.is_valid (bool) & frame.quality.classification (str)
Frame Timestamp:         frame.timestamp (monotonic float in seconds)
```

---

## 3. Downstream Temporal Window Requirements

| Parameter | Specification | Purpose in ST-GCN Inference |
| :--- | :--- | :--- |
| **Sequence Length ($T$)** | Exactly 45 frames (1.8 seconds) | Fixed temporal input dimension of `SignTalk_STGCN_v1` |
| **Target Frame Rate** | 25.0 FPS | Matches temporal dynamics of training dataset |
| **Tensor Geometry** | `[B, C, T, V] = [1, 3, 45, 93]` | Direct batch input to ST-GCN backbone |
| **Sliding Window Stride** | 5 to 10 frames (200 – 400 ms update rate) | Smooth real-time prediction cadence without excessive compute |
| **Missing Frame Bridge** | 1D temporal linear interpolation | Bridges transient camera drops (up to 10 frames) to prevent ST-GCN accuracy collapse |

---

## 4. Recommended Sequence Buffer Implementation for Part 2

```python
class TemporalSequenceBuffer:
    """
    Sliding window ring buffer accumulating real-time LandmarkFrames into
    model-ready tensors [1, 3, 45, 93].
    """
    def __init__(self, target_length: int = 45, stride: int = 5):
        self.target_length = target_length
        self.stride = stride
        self.buffer = collections.deque(maxlen=target_length)
        self.frames_since_last_inference = 0

    def append(self, frame: LandmarkFrame) -> Optional[torch.Tensor]:
        self.buffer.append(frame)
        self.frames_since_last_inference += 1

        if len(self.buffer) == self.target_length and self.frames_since_last_inference >= self.stride:
            self.frames_since_last_inference = 0
            return self._build_model_tensor()
        return None

    def _build_model_tensor(self) -> torch.Tensor:
        # Stack 45 frames of (93, 3) -> (45, 93, 3)
        coords = np.stack([f.normalized_coords for f in self.buffer], axis=0)
        # Transpose to (3, 45, 93) [C, T, V]
        tensor_c_t_v = np.transpose(coords, (2, 0, 1))
        # Batch dimension: [1, 3, 45, 93]
        return torch.from_numpy(tensor_c_t_v).float().unsqueeze(0)
```

---

## 5. Measured Performance Baseline

* **Ingestion Latency (CPU):** ~133 ms mean / ~139 ms median (Extraction: ~132 ms, Preprocessing + Normalization + Quality: < 1.0 ms)
* **Ingestion Throughput (CPU):** ~8.4 FPS (CPU XNNPACK delegate)
* **Target Optimization for Part 2 / Edge:**
  * MediaPipe GPU / OpenCL delegate acceleration or lightweight detector models to bring extraction latency $< 25\text{ ms}$.
  * Export ST-GCN backbone to ONNX Runtime CPU (`ExecutionProvider='CPUExecutionProvider'`) to reduce inference latency from ~80 ms to $< 20\text{ ms}$.

---

## 6. Known Limitations to Address in Part 2

1. **Signer Quality Gate:** Reject windows with $< 15$ valid frames or where hands are not present in the signing volume before invoking ST-GCN forward pass.
2. **Confidence Rejection Threshold:** Enforce $\tau = 0.70$ rejection threshold to ensure 0% false acceptances as established in Phase 3 Part 4.
3. **Prediction Stabilization:** Apply rolling voting across 3 consecutive inference windows to prevent flickering labels in the user interface.
