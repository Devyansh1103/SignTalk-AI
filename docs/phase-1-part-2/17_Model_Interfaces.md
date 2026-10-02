# 17. Code-Level Model Interfaces & Type Signatures: SignTalk AI

**Document ID:** STAI-P1P2-017  
**Project Name:** SignTalk AI  
**Document Status:** Approved Engineering Blueprint (Phase 1 — Part 2)  

---

## 1. Type Definitions & Dataclasses

To ensure rigorous type safety across scientific and web subsystems, all internal data exchange structures are formalized as Python `dataclass` and `Pydantic` schemas:

```python
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch

@dataclass(frozen=True)
class NormalizedFrame:
    """Represents a single normalized spatial frame of 93 landmarks."""
    frame_id: int
    timestamp_ms: int
    coordinates: np.ndarray        # Shape: (93, 3), dtype: np.float32
    visibility: np.ndarray         # Shape: (93,), dtype: np.float32
    left_hand_present: bool
    right_hand_present: bool

@dataclass
class GraphTensorBundle:
    """Represents an assembled spatial-temporal graph ready for ST-GCN forward pass."""
    x: torch.Tensor                # Shape: (B, C_in, T, N) = (1, 3, 45, 93), dtype: torch.float32
    a_norm: torch.Tensor           # Shape: (K, N, N) = (3, 93, 93), dtype: torch.float32
    mask: Optional[torch.Tensor]   # Shape: (B, T) = (1, 45), dtype: torch.bool

@dataclass
class TranslationOutput:
    """Represents the decoded natural-language sentence and safety confidence."""
    text: str
    confidence: float              # Calibrated score in [0.0, 1.0]
    token_ids: List[int]
    is_valid: bool                 # True if confidence >= 0.50
    stgcn_latency_ms: float
    transformer_latency_ms: float
```

---

## 2. Abstract Subsystem Protocols

```python
from typing import Protocol

class LandmarkExtractorProtocol(Protocol):
    def extract_landmarks(self, frame_bgr: np.ndarray) -> Optional[NormalizedFrame]:
        """Extracts and normalizes 93 3D keypoints from an input BGR frame."""
        ...

class GraphBuilderProtocol(Protocol):
    def build_graph(self, window_frames: List[NormalizedFrame]) -> GraphTensorBundle:
        """Assembles a temporal sequence of normalized frames into an ST-GCN graph tensor."""
        ...

class STGCNEncoderProtocol(Protocol):
    def forward(self, bundle: GraphTensorBundle) -> torch.Tensor:
        """Processes graph tensor, returning latent sequence tokens H of shape (B, T', d_model)."""
        ...

class TransformerDecoderProtocol(Protocol):
    def decode(self, memory: torch.Tensor, max_len: int = 20) -> Tuple[List[int], List[float]]:
        """Autoregressively decodes latent memory into token IDs and likelihoods."""
        ...

class TokenizerProtocol(Protocol):
    def detokenize(self, token_ids: List[int]) -> str:
        """Converts discrete token IDs into formatted English text."""
        ...
```

---

## 3. Exception Hierarchy & Failure Mitigation

| Exception Class | Trigger Condition | System Mitigation |
| :--- | :--- | :--- |
| **`TrackingLostException`** | Both hands unobserved ($c_{vis} < 0.25$) for $\ge 15$ consecutive frames. | Bypasses ST-GCN forward; emits `IDLE` state; prompts user: *"Please position hands in camera frame."* |
| **`BufferUnderflowException`** | Ring buffer contains fewer than $T_{min} = 15$ frames upon session startup. | Suppresses model inference until buffer accumulates sufficient temporal context. |
| **`ModelInferenceTimeout`** | ST-GCN or Transformer forward execution exceeds $300\text{ ms}$. | Aborts ongoing forward pass; logs thermal/compute warning; falls back to greedy single-token decoding. |
| **`OutOfVocabularyException`**| Model emits unknown tokens or punctuation syntax errors. | Post-processor filters `<UNK>` tokens and strips dangling punctuation. |
