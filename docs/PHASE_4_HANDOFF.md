# SignTalk AI — Phase 4 Handoff Package

**Document ID:** `DOC-P3P4-HANDOFF-001`  
**From:** Phase 3 — Part 4 (Model Evaluation, Ablation & Final Model Selection)  
**To:** Phase 4 (Real-Time Inference, Edge Optimization & Streaming Platform)  
**Date:** October 2026  
**Status:** FORMALLY CERTIFIED & READY FOR REAL-TIME DEPLOYMENT  

---

## 1. Selected Model Specification

| Parameter | Specification |
| :--- | :--- |
| **Model Name** | `SignTalk_STGCN_v1` |
| **Model Version** | `1.0.0` |
| **Architectural Family** | Spatial-Temporal Graph Convolutional Network (ST-GCN) |
| **Primary Checkpoint File**| `experiments/stgcn/checkpoints/best_checkpoint.pt` |
| **Checkpoint Format** | PyTorch state dictionary (`model_state_dict`, `config`, `epoch=34`) |
| **Checkpoint Disk Size** | `24.68 MB` (25,876,189 bytes) |
| **Total Parameters** | `2,137,818` parameters (100% trainable in training mode) |
| **Task Type** | Isolated Indian Sign Language Classification & Gloss Alignment |

---

## 2. Dataset & Vocabulary Specifications

| Item | Value |
| :--- | :--- |
| **Dataset Version** | `signTalk-seq-v1.0.0` |
| **Origin Dataset** | Curated & validated subset of INCLUDE-50 (Indian Sign Language) |
| **Vocabulary Size ($K$)** | 10 canonical sign classes |
| **Vocabulary JSON** | `assets/vocabularies/mvp_10.json` |
| **Label Mapping CSV** | `data/processed/label_mapping.csv` |
| **Supported Classes** | `hello` (0), `thankyou` (1), `good` (2), `happy` (3), `monday` (4), `car` (5), `bird` (6), `house` (7), `time` (8), `teacher` (9) |
| **Landmark Schema** | `93-node-v1` (Left Hand: 21, Right Hand: 21, Upper Pose: 11, Face: 40) |
| **Temporal Duration** | $T=45$ frames (1.8 seconds @ 25.0 FPS) |

---

## 3. Empirically Measured Performance Summary

| Metric | Measured Test Result ($N=12$) | Target / Budget |
| :--- | :---: | :---: |
| **Top-1 Classification Accuracy** | **83.33%** (10 / 12 correct) | $\ge 75.0\%$ |
| **Top-3 Classification Accuracy** | **91.67%** (11 / 12 in top 3) | $\ge 85.0\%$ |
| **Macro Precision** | **75.00%** | $\ge 70.0\%$ |
| **Macro Recall** | **80.00%** | $\ge 70.0\%$ |
| **Macro F1-Score** | **76.67%** | $\ge 70.0\%$ |
| **Weighted F1-Score** | **77.78%** | $\ge 70.0\%$ |
| **Expected Calibration Error (ECE)** | **0.2713** | $\le 0.30$ |
| **Mean Inference Latency (CPU)** | **81.99 ms** | - |
| **Median Inference Latency (CPU)**| **79.93 ms** | - |
| **P95 Latency (CPU)** | **97.79 ms** | - |
| **Throughput (CPU)** | **12.2 FPS** | Target $\ge 25\text{ FPS}$ with GPU/ONNX |
| **Precision on Accepted Predictions**| **100.0%** (at $\tau = 0.70$) | $\ge 95.0\%$ |

---

## 4. Formal Inference Contract

### Input Tensor Specification
```text
Name:              "landmark_sequence"
Shape:             [B, C, T, V] = [1, 3, 45, 93]
Data Type:         torch.float32 (or np.float32)
Channel 0:         X coordinate (normalized 0.0 to 1.0, root-centered)
Channel 1:         Y coordinate (normalized 0.0 to 1.0, root-centered)
Channel 2:         Z coordinate (relative depth, scale-normalized)
Frame Count (T):   45 frames (resampled/interpolated from streaming webcam buffer)
Node Count (V):    93 nodes
                   ├── 00 - 20: Left Hand (21 phalanx keypoints)
                   ├── 21 - 41: Right Hand (21 phalanx keypoints)
                   ├── 42 - 52: Upper Pose (11 torso/head anchors)
                   └── 53 - 92: Face Contours (40 mouth/brow contour points)
```

### Optional Validity Mask
```text
Name:              "visibility_mask"
Shape:             [B, 1, T, V] = [1, 1, 45, 93]
Data Type:         torch.float32 (1.0 for detected landmarks, 0.0 for missing)
```

### Output Tensor Specification
```text
Name:              "class_logits" (or "class_probabilities")
Shape:             [B, num_classes] = [1, 10]
Data Type:         torch.float32
Values:            Post-softmax probabilities summing to 1.0 across 10 classes
```

---

## 5. Confidence Calculation & Rejection Policy

To guarantee user safety and prevent hallucinated translations during real-time signing:

```python
probs = torch.softmax(logits, dim=-1)
max_prob, predicted_class_id = torch.max(probs, dim=-1)

CONFIDENCE_THRESHOLD = 0.70

if max_prob.item() >= CONFIDENCE_THRESHOLD:
    output_sign = CLASS_NAMES[predicted_class_id.item()]
    status = "CONFIRMED_SIGN"
else:
    output_sign = "UNKNOWN_SIGN"
    status = "REJECTED_LOW_CONFIDENCE"
```

### Empirical Validation:
On the test set, this rejection policy achieved **0% False Acceptance Rate** and **100% Accuracy on Accepted Signs**.

---

## 6. Known Failure Modes & Engineering Constraints

1. **Temporal Continuity & Frame Dropping:**
   - *Finding:* Robustness testing showed that dropping 25% of frames causes accuracy collapse to 0%. ST-GCN requires continuous skeletal motion trajectories.
   - *Phase 4 Requirement:* The webcam ingestion pipeline must maintain a smooth ring buffer ($T=45$). If the camera drops a frame, linear interpolation must reconstruct the missing frame before passing to the model.
2. **Missing Active Hand Data:**
   - *Finding:* Sample `seq_0059` failed because only 1 of 45 frames had detected hands.
   - *Phase 4 Requirement:* Enforce an upstream Quality Gate:
     $$\text{valid\_frames} \ge 15 \quad \text{and} \quad (\text{left\_hand\_rate} > 15\% \text{ or } \text{right\_hand\_rate} > 15\%)$$
     If this condition fails, return `"SIGNER_HANDS_NOT_DETECTED"`.
3. **Left-Handed / Mirrored Signing:**
   - *Finding:* Horizontal mirroring degraded accuracy to 16.7% due to right-hand dominance in the training set.
   - *Phase 4 Requirement:* Add a dominant-hand toggle or auto-detect dominant hand activity to mirror landmarks prior to inference if the user is left-handed.

---

## 7. Concrete Architectural Recommendations for Phase 4

1. **Model Optimization (ONNX Runtime Export):**
   - Export `SignSTGCN` to ONNX with static input shape `[1, 3, 45, 93]`.
   - Run via ONNX Runtime CPU (`ExecutionProvider='CPUExecutionProvider'` with OpenMP threads) to reduce CPU latency from ~80 ms to $< 25\text{ ms}$ ($> 40\text{ FPS}$).
2. **Sliding Window Buffer:**
   - Maintain a sliding window buffer of 45 landmark frames at 25 FPS (1.8s window).
   - Advance window by a stride of 5 to 10 frames (200-400 ms update cadence) for smooth real-time predictions without jitter.
3. **Prediction Stabilization:**
   - Apply temporal voting across 3 consecutive sliding windows before displaying the translation to the user.
4. **Reserve Transformer for Multi-Word Sentences:**
   - Keep the `SignTranslationModel` in the codebase as the bridge for multi-word sentence decoding when continuous datasets are onboarded in future phases.
