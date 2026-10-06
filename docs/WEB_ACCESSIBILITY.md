# SignTalk AI: Web Accessibility & Inclusive Design

**Project**: SignTalk AI: Real-Time Indian Sign Language Translation and Communication Platform  
**Guideline Compliance**: WCAG 2.1 Level AA  
**Phase**: Phase 5  

---

## 1. Principles of Inclusive Design for Sign Language Users

Designing an interface for Indian Sign Language (ISL) users requires specific visual, ergonomic, and temporal considerations that differ from standard audio-centric or text-only interfaces:

1. **Zero Reliance on Audio**: All notifications, pipeline states, and recognition feedbacks are purely visual.
2. **Visual Ergonomics**: The signer stands or sits at arm's length (60–100 cm) from the camera and screen. Text and controls must be legible from a distance.
3. **Low Visual Fatigue**: Color schemes avoid pure neon or oversaturated primaries; dark mode reduces retinal glare during extended signing sessions.
4. **Immediate Continuous Feedback**: Signers require instant visual confirmation that their hand gestures are within the camera frame without glancing away from the lens.

---

## 2. Accessibility Modes

### 2.1 High Contrast Mode
- **Activation**: Toggle switch in header / footer (`AccessibilityControls.tsx`).
- **Styling**: Enforces absolute black backgrounds (`#000000`), crisp pure-white borders (`#ffffff`), and high-luminance text accents.
- **Contrast Ratios**: Meets and exceeds WCAG AAA criteria ($> 7:1$ contrast ratio) for body text and visual UI controls.

### 2.2 Large Text Mode
- **Activation**: Toggle switch in header (`AccessibilityControls.tsx`).
- **Typography Scale**:
  - Live Sign Gloss: Scales from `text-3xl` (30px) to `text-5xl` (48px).
  - Translated English Phrase: Scales to `text-2xl` (24px) bold.
  - Translated Hindi Phrase: Scales to `text-3xl` (30px) Devanagari script.
- **Distance Legibility**: Legible from 2 meters away on a standard 14-inch laptop display.

---

## 3. Keyboard Navigation & Assistive Switch Support

For users utilizing assistive foot pedals, single-switch devices, or standard keyboards, all primary session controls are operable without mouse interaction:

| Shortcut | Target Action | Behavior |
| :---: | :--- | :--- |
| `Space` | Start / Pause | Toggles camera streaming and inference engine |
| `R` or `r` | Reset Session | Flushes temporal buffer, transcript, and sequence state |
| `S` or `s` | Developer HUD | Toggles latency telemetry overlay |
| `Tab` | Focus Traversal | Logical, linear focus order across all interactive controls |
| `Enter` / `Space` | Activate Element | Triggers focused buttons and modals |
| `Escape` | Close Modal | Closes Vocabulary and Privacy dialogs |

All focusable elements feature prominent, high-visibility focus rings (`focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:outline-none`).

---

## 4. Screen Reader Semantics & ARIA Patterns

The application conforms to modern WAI-ARIA standards for assistive technologies:

1. **Live Translation Announcement**:
   - The primary translation container includes `aria-live="polite"` and `role="status"`.
   - As new sign phrases finalize, screen readers announce the translation without interrupting prior speech or dominating focus.
2. **Camera Guides**:
   - Camera viewport elements include descriptive `aria-label` attributes (e.g., `aria-label="Webcam view with real-time pose and hand landmark tracking"`).
3. **Modal Dialogs**:
   - The Vocabulary and Privacy modals implement standard `role="dialog"`, `aria-modal="true"`, and `aria-labelledby` attributes with background backdrop blurring and keyboard traps.

---

## 5. Visual Framing & Ergonomic Guides

The camera viewport includes an embedded [CameraGuide.tsx](file:///d:/SignAI/frontend/src/components/camera/CameraGuide.tsx) overlay that visually assists the signer:
- Silhouette boundary indicates optimal signing space (head, shoulders, and chest area).
- Real-time indicator notifies when hands move outside the high-confidence camera frustum.
- Eliminates guesswork regarding webcam distance and elevation.
