# SignTalk AI: Phase 5 — Web Application & User Experience

**Project**: SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Technical Subtitle**: A Spatial-Temporal Graph and Transformer-Based Sign-to-Text System  
**Phase**: Phase 5 (Web Application & User Experience)  
**Status**: Certified & Complete  

---

## 1. Executive Overview

Phase 5 transitions the certified real-time inference engine from **Phase 4** into a responsive, accessible, high-performance web application designed for both Deaf/Hard-of-Hearing individuals and hearing communication partners.

### Core Value Proposition
- **Real-Time Visual Translation**: Sub-100ms spatial-temporal inference stream displaying recognized signs, live bilingual captions (English + Hindi), and temporal stability.
- **Accessibility by Design**: High-contrast mode, dynamic font scaling (Large Text Mode), keyboard-accessible control deck, screen-reader friendly `aria-live` captioning, and non-blocking visual telemetry.
- **Product Truthfulness & Transparency**: Honest presentation of the certified vocabulary set (**MVP-10**: `hello`, `thankyou`, `good`, `happy`, `monday`, `car`, `bird`, `house`, `time`, `teacher`) with no overclaiming of general ISL vocabulary or unrestricted conversational translation.
- **Privacy First**: Zero video or landmark data stored remotely; real-time video frames are processed in-memory for inference and discarded instantly.

---

## 2. System Architecture

The web application follows a decoupled client-server architecture connected via high-throughput bidirectional WebSockets:

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Browser                         │
│  ┌───────────────────────┐       ┌───────────────────────┐  │
│  │    HTML5 Camera API   │       │  React 19 + Vite UI   │  │
│  │ (Canvas Frame Grab)   │       │  (Lucide + Tailwind)  │  │
│  └──────────┬────────────┘       └───────────▲───────────┘  │
│             │                                │              │
│             │ Base64 JPEG frames             │ JSON Events  │
│             ▼                                │              │
│     WebSocket Client (useRealtime / RealtimeClient)         │
└─────────────────────┬────────────────────────▲──────────────┘
                      │                        │
               ws://localhost:8000/ws/realtime │
                      │                        │
┌─────────────────────▼────────────────────────┴──────────────┐
│                    FastAPI Backend Server                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │            WebSocket Connection Router                │  │
│  │       (backend/app/api/realtime.py)                   │  │
│  └──────────────────────────┬────────────────────────────┘  │
│                             │                               │
│  ┌──────────────────────────▼────────────────────────────┐  │
│  │             InferenceService Adapter                  │  │
│  │       (backend/app/services/inference_service.py)     │  │
│  └──────────────────────────┬────────────────────────────┘  │
│                             │                               │
│  ┌──────────────────────────▼────────────────────────────┐  │
│  │       RealtimePipeline (Phase 4 Certified Core)       │  │
│  │ ┌──────────────┐ ┌──────────────┐ ┌────────────────┐  │  │
│  │ │  MediaPipe   │ │   ST-GCN     │ │   Linguistic   │  │  │
│  │ │ Landmark Ext │ │ 16-Frame Win │ │  Translator    │  │  │
│  │ └──────────────┘ └──────────────┘ └────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Frontend Component Breakdown

The frontend application is structured into modular components in `frontend/src/components/`:

### A. Layout (`components/layout/`)
- [Header.tsx](file:///d:/SignAI/frontend/src/components/layout/Header.tsx): Displays project branding, active model identifier (`SignTalk_STGCN_v1`), connection status badge, and accessibility quick-actions.
- [Footer.tsx](file:///d:/SignAI/frontend/src/components/layout/Footer.tsx): Displays privacy pledge (zero video storage), vocabulary modal trigger, and engineering attribution.

### B. Camera Viewport (`components/camera/`)
- [CameraView.tsx](file:///d:/SignAI/frontend/src/components/camera/CameraView.tsx): Renders the mirrored webcam video feed with landmark tracking overlays, signer boundary guides, FPS counter, and device selector.
- [CameraGuide.tsx](file:///d:/SignAI/frontend/src/components/camera/CameraGuide.tsx): Ergonomic framing overlay instructing users on optimal posture (hands within frame, shoulder distance, neutral lighting).

### C. Translation Deck (`components/translation/`)
- [CurrentSign.tsx](file:///d:/SignAI/frontend/src/components/translation/CurrentSign.tsx): Prominently highlights the currently detected and stabilized sign with a color-coded confidence tier badge.
- [ConfidenceIndicator.tsx](file:///d:/SignAI/frontend/src/components/translation/ConfidenceIndicator.tsx): Accessible visual meter with discrete tiers (`high` $\ge 0.70$, `medium` $\ge 0.50$, `low` $\ge 0.40$, `uncertain` $< 0.40$).
- [TranslationPanel.tsx](file:///d:/SignAI/frontend/src/components/translation/TranslationPanel.tsx): Bilingual translated text card showing English phrase, Hindi (Devanagari script) equivalent, sign sequence tags, and quick-copy clipboard button.
- [LiveTranscript.tsx](file:///d:/SignAI/frontend/src/components/translation/LiveTranscript.tsx): Chronological log of finalized communication phrases with timestamps, clear-transcript functionality, and empty state guidance.

### D. Controls & Telemetry (`components/controls/`, `components/debug/`, `components/status/`)
- [SessionControls.tsx](file:///d:/SignAI/frontend/src/components/controls/SessionControls.tsx): Primary interaction dock with Start, Pause/Resume, and Reset actions alongside keyboard shortcut reminders.
- [AccessibilityControls.tsx](file:///d:/SignAI/frontend/src/components/controls/AccessibilityControls.tsx): Toggle switches for High Contrast Mode and Large Text Mode.
- [DeveloperHUD.tsx](file:///d:/SignAI/frontend/src/components/debug/DeveloperHUD.tsx): Real-time telemetry inspector rendering pipeline stage latencies (MediaPipe, sliding window, ST-GCN, translation, end-to-end), buffer occupancy, and FPS.
- [ConnectionBadge.tsx](file:///d:/SignAI/frontend/src/components/status/ConnectionBadge.tsx): Visual status pill with pulsing indicator (`connected`, `processing`, `paused`, `disconnected`, `error`).

### E. Modals (`components/modals/`)
- [VocabularyModal.tsx](file:///d:/SignAI/frontend/src/components/modals/VocabularyModal.tsx): Interactive catalog of all 10 supported signs with English glosses, Hindi translations, and signer tips.
- [PrivacyModal.tsx](file:///d:/SignAI/frontend/src/components/modals/PrivacyModal.tsx): Transparent disclosure of data protection policies confirming zero biometric archiving.

---

## 4. Key UX & Accessibility Features

| Feature | Implementation | Accessibility Benefit |
| :--- | :--- | :--- |
| **High Contrast Mode** | Black background (`#000000`), pure white borders (`#ffffff`), enhanced visual contours | Optimizes readability for users with low vision or photophobia |
| **Large Text Mode** | Increases translation typography to `text-3xl`/`text-4xl` and gloss labels | Allows effortless readability at arm's length while signing |
| **ARIA Live Captions** | `<div aria-live="polite" role="status">` | Enables screen readers to announce new translations without interrupting user speech |
| **Keyboard Deck** | `Space` (Start/Pause), `R` (Reset), `S` (Toggle HUD) | Enables hands-free control via accessible pedals or switches |
| **Uncertainty Feedback** | Visual amber badge with "Uncertain - hold sign clearly" prompt | Eliminates signer confusion during ambiguous transitions |

---

## 5. Technology Stack

- **Frontend Runtime**: React 19.2, TypeScript 5.8 (Vite 8.3)
- **Styling**: Tailwind CSS v4 (`@tailwindcss/vite`) with custom responsive tokens
- **Icons**: Lucide React (`lucide-react`)
- **Backend API**: FastAPI (Python 3.13) with Starlette WebSockets & Pydantic v2
- **Inference Runtime**: PyTorch 2.13.0 CPU inference with MediaPipe 0.10.32
- **Testing**: PyTest 9.1 for backend integration; Vite build verification for frontend type safety
