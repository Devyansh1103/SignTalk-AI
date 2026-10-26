/**
 * SignTalk AI - In-browser sign pipeline (port of the Python live backend, segmentation mode).
 *
 * Mirrors, step for step:
 *   src/realtime/landmark_stream.py   per-frame torso normalization (EMA anchor) + quality
 *   src/realtime/sign_segmenter.py    hands rise -> hands drop segmentation, resample_by_time
 *   src/realtime/model_runner.py      offline post-process (gap interpolation, window normalization)
 *   src/realtime/realtime_pipeline.py accept/reject, display state, sign sequence
 *   src/realtime/translator.py        English / Hindi phrases with pause finalization
 * Settings come from public/models/pipeline.json (written by scripts/export_onnx.py from
 * configs/realtime.yaml). No browser APIs here, so the parity test can run it under Node.
 */

export const NODES = 93;

export interface PipelineSettings {
  normalization: {
    min_scale_epsilon: number;
    z_scale_factor: number;
    use_temporal_anchor_smoothing: boolean;
    anchor_smoothing_alpha: number;
    reference_aspect: number;
  };
  quality: { weights: Record<string, number>; good: number; acceptable: number };
  segmentation: {
    active_max_y: number;
    min_active_sec: number;
    end_hold_sec: number;
    pad_before_sec: number;
    pad_after_sec: number;
    max_segment_sec: number;
    min_confidence: number;
    min_margin: number;
  };
  sequence_length: number;
  max_consecutive_missing: number;
  translation: { pause_threshold_sec: number; min_confidence: number; max_phrase_signs: number };
  max_events: number;
  top_k: number;
}

export interface VocabEntry {
  class_id: number;
  label: string;
  gloss: string;
  english_translation: string;
  hindi_translation: string;
}

/** One frame of raw MediaPipe output in the 93-node layout (see landmarks.ts). */
export interface RawLandmarks {
  coords: Float32Array; // 93 * 3, image-normalized x, y, z
  mask: Uint8Array; // 93
  visibility: Float32Array; // 93
  poseDetected: boolean;
  leftHandDetected: boolean;
  rightHandDetected: boolean;
  poseConfidence: number;
  leftHandConfidence: number;
  rightHandConfidence: number;
}

export interface LandmarkFrame {
  timestamp: number; // seconds
  raw: RawLandmarks;
  normalized: Float32Array; // 93 * 3
  aspectYScale: number;
  qualityScore: number;
  isValid: boolean;
}

export interface ModelInput {
  x: Float32Array; // [1, 3, T, 93]
  mask: Float32Array; // [1, 1, T, 93]
}

interface SignEvent {
  label: string;
  confidence: number;
  startTime: number;
  endTime: number;
}

export interface Translation {
  text: string;
  hindi_text: string;
  confidence: number;
  tokens: string[];
  source_signs: string[];
  timestamp: number;
  status: string;
  is_final: boolean;
}

export interface Prediction {
  class_id: number;
  label: string;
  gloss: string;
  confidence: number;
  input_quality: number;
  top_k: [number, string, number][];
  accepted: boolean;
}

// --------------------------------------------------------------------------------------------
// Per-frame normalization and quality (landmark_stream.py)
// --------------------------------------------------------------------------------------------

/** Torso anchor of one frame: mid-shoulder and shoulder width, with the same fallbacks. */
function frameAnchor(c: ArrayLike<number>, m: ArrayLike<number>, eps: number): { center: number[]; scale: number } {
  const ls = m[47], rs = m[48];
  if (ls && rs) {
    const center = [0, 1, 2].map((k) => (c[47 * 3 + k] + c[48 * 3 + k]) / 2);
    const d = Math.hypot(c[47 * 3] - c[48 * 3], c[47 * 3 + 1] - c[48 * 3 + 1], c[47 * 3 + 2] - c[48 * 3 + 2]);
    return { center, scale: Math.max(d, eps) };
  }
  if (ls) return { center: [c[141], c[142], c[143]], scale: 0.25 };
  if (rs) return { center: [c[144], c[145], c[146]], scale: 0.25 };
  const pose: number[] = [];
  for (let v = 42; v < 53; v++) if (m[v]) pose.push(v);
  if (pose.length) {
    const center = [0, 1, 2].map((k) => pose.reduce((s, v) => s + c[v * 3 + k], 0) / pose.length);
    return { center, scale: 0.25 };
  }
  return { center: [0.5, 0.5, 0], scale: 1 };
}

function applyAnchor(c: ArrayLike<number>, m: ArrayLike<number>, center: number[], scale: number, z: number, out: Float32Array, off = 0) {
  for (let v = 0; v < NODES; v++) {
    const o = off + v * 3;
    if (m[v]) {
      out[o] = (c[v * 3] - center[0]) / scale;
      out[o + 1] = (c[v * 3 + 1] - center[1]) / scale;
      out[o + 2] = ((c[v * 3 + 2] - center[2]) / scale) * z;
    } else {
      out[o] = out[o + 1] = out[o + 2] = 0;
    }
  }
}

// --------------------------------------------------------------------------------------------
// Segmentation (sign_segmenter.py)
// --------------------------------------------------------------------------------------------

export function resampleByTime<T extends { timestamp: number }>(frames: T[], t0: number, t1: number, n: number): T[] {
  const ts = frames.map((f) => f.timestamp);
  const out: T[] = [];
  for (let i = 0; i < n; i++) {
    const target = n === 1 ? t0 : t0 + ((t1 - t0) * i) / (n - 1);
    let hi = ts.findIndex((t) => t >= target); // np.searchsorted (left)
    if (hi < 0) hi = ts.length;
    hi = Math.min(Math.max(hi, 1), ts.length - 1);
    const lo = hi - 1;
    out.push(frames[Math.abs(target - ts[lo]) <= Math.abs(ts[hi] - target) ? lo : hi]);
  }
  return out;
}

class SignSegmenter {
  private history: LandmarkFrame[] = [];
  private inSeg = false;
  private run = 0;
  private runStart = 0;
  private segStart = 0;
  private lastActive = 0;
  private readonly historySec: number;
  private readonly minStartFrames = 2;
  private readonly s: PipelineSettings['segmentation'];

  constructor(s: PipelineSettings['segmentation']) {
    this.s = s;
    this.historySec = Math.max(7.0, s.max_segment_sec + s.pad_before_sec + s.pad_after_sec + 1.0);
  }

  get inSegment() {
    return this.inSeg;
  }

  reset() {
    this.history = [];
    this.inSeg = false;
    this.run = 0;
    this.runStart = this.segStart = this.lastActive = 0;
  }

  private isActive(f: LandmarkFrame): boolean {
    let minY = Infinity;
    for (let v = 0; v < 42; v++) if (f.raw.mask[v]) minY = Math.min(minY, f.normalized[v * 3 + 1]);
    return minY !== Infinity && minY <= this.s.active_max_y;
  }

  update(f: LandmarkFrame): LandmarkFrame[] | null {
    const ts = f.timestamp;
    this.history.push(f);
    while (this.history.length && this.history[0].timestamp < ts - this.historySec) this.history.shift();

    const active = this.isActive(f);
    if (!this.inSeg) {
      if (active) {
        if (this.run === 0) this.runStart = ts;
        this.run += 1;
        if (this.run >= this.minStartFrames) {
          this.inSeg = true;
          this.segStart = this.runStart;
          this.lastActive = ts;
        }
      } else {
        this.run = 0;
      }
      return null;
    }

    if (active) this.lastActive = ts;
    if (ts - this.lastActive >= this.s.end_hold_sec) {
      this.inSeg = false;
      this.run = 0;
      if (this.lastActive - this.segStart < this.s.min_active_sec) return null;
      return this.slice(this.segStart - this.s.pad_before_sec, this.lastActive + this.s.pad_after_sec);
    }
    if (ts - this.segStart >= this.s.max_segment_sec) {
      const seg = this.slice(this.segStart - this.s.pad_before_sec, ts);
      this.segStart = ts;
      return seg;
    }
    return null;
  }

  private slice(t0: number, t1: number): LandmarkFrame[] | null {
    const frames = this.history.filter((f) => f.timestamp >= t0 && f.timestamp <= t1);
    return frames.length >= 2 ? frames : null;
  }
}

// --------------------------------------------------------------------------------------------
// Model input (model_runner._offline_postprocess + sequence_utils.handle_missing_landmarks)
// --------------------------------------------------------------------------------------------

export function windowToInput(window: LandmarkFrame[], s: PipelineSettings): ModelInput {
  const T = window.length;
  const coords = new Float32Array(T * NODES * 3);
  const mask = new Uint8Array(T * NODES);
  window.forEach((f, t) => {
    for (let v = 0; v < NODES; v++) {
      const o = (t * NODES + v) * 3;
      coords[o] = f.raw.coords[v * 3];
      coords[o + 1] = f.raw.coords[v * 3 + 1] * f.aspectYScale;
      coords[o + 2] = f.raw.coords[v * 3 + 2];
      mask[t * NODES + v] = f.raw.mask[v];
    }
  });

  // Linear interpolation over interior gaps of up to max_consecutive_missing frames. Exterior and
  // longer gaps stay unmasked, and unmasked points are zeroed below, as in Python.
  const maxGap = s.max_consecutive_missing;
  for (let v = 0; v < NODES; v++) {
    let prev = -1;
    for (let t = 0; t < T; t++) {
      if (!mask[t * NODES + v]) continue;
      const gap = t - prev - 1;
      if (prev >= 0 && gap > 0 && gap <= maxGap) {
        for (let step = 1; step <= gap; step++) {
          const a = step / (gap + 1), q = ((prev + step) * NODES + v) * 3;
          for (let k = 0; k < 3; k++) {
            coords[q + k] = (1 - a) * coords[(prev * NODES + v) * 3 + k] + a * coords[(t * NODES + v) * 3 + k];
          }
          mask[(prev + step) * NODES + v] = 1;
        }
      }
      prev = t;
    }
  }

  // Window-wide torso normalization (model_runner._normalize_window)
  const n = s.normalization;
  const out = new Float32Array(T * NODES * 3);
  let cx = 0, cy = 0, cz = 0, dist = 0, both = 0;
  for (let t = 0; t < T; t++) {
    if (!(mask[t * NODES + 47] && mask[t * NODES + 48])) continue;
    const l = (t * NODES + 47) * 3, r = (t * NODES + 48) * 3;
    cx += (coords[l] + coords[r]) / 2;
    cy += (coords[l + 1] + coords[r + 1]) / 2;
    cz += (coords[l + 2] + coords[r + 2]) / 2;
    dist += Math.hypot(coords[l] - coords[r], coords[l + 1] - coords[r + 1], coords[l + 2] - coords[r + 2]);
    both += 1;
  }
  for (let t = 0; t < T; t++) {
    const c = coords.subarray(t * NODES * 3, (t + 1) * NODES * 3);
    const m = mask.subarray(t * NODES, (t + 1) * NODES);
    if (both) {
      applyAnchor(c, m, [cx / both, cy / both, cz / both], Math.max(dist / both, n.min_scale_epsilon), n.z_scale_factor, out, t * NODES * 3);
    } else {
      // No frame with both shoulders: per-frame anchors (CoordinateNormalizer.normalize_frame)
      const a = frameAnchor(c, m, n.min_scale_epsilon);
      applyAnchor(c, m, a.center, a.scale, n.z_scale_factor, out, t * NODES * 3);
    }
  }

  // (T, V, C) -> (1, C, T, V)
  const x = new Float32Array(3 * T * NODES);
  const mk = new Float32Array(T * NODES);
  for (let t = 0; t < T; t++) {
    for (let v = 0; v < NODES; v++) {
      for (let k = 0; k < 3; k++) x[k * T * NODES + t * NODES + v] = out[(t * NODES + v) * 3 + k];
      mk[t * NODES + v] = mask[t * NODES + v];
    }
  }
  return { x, mask: mk };
}

// --------------------------------------------------------------------------------------------
// Translation (translator.py)
// --------------------------------------------------------------------------------------------

const PHRASE_TEMPLATES: Record<string, [string, string]> = {
  'hello teacher': ['Hello teacher!', 'नमस्ते शिक्षक!'],
  'good monday': ['Good Monday!', 'शुभ सोमवार!'],
  'happy monday': ['Happy Monday!', 'शुभ सोमवार!'],
  'good teacher': ['Good teacher.', 'अच्छे शिक्षक।'],
  'happy bird': ['Happy bird.', 'खुश पक्षी।'],
  'hello happy': ['Hello, I am happy.', 'नमस्ते, मैं खुश हूँ।'],
  'thankyou teacher': ['Thank you, teacher.', 'धन्यवाद, शिक्षक।'],
  'time house': ['Time to go home.', 'घर जाने का समय।'],
  'car house': ['The car is at the house.', 'गाड़ी घर पर है।'],
  'bird house': ['A bird is near the house.', 'घर के पास पक्षी है।'],
  'good time': ['Have a good time.', 'अच्छा समय बीते।'],
};

class Translator {
  private active: SignEvent[] = [];
  private lastEventTime: number | null = null;
  private readonly pauseSec: number;
  private readonly t: PipelineSettings['translation'];
  private readonly english: Map<string, string>;
  private readonly hindi: Map<string, string>;

  constructor(t: PipelineSettings['translation'], vocab: VocabEntry[]) {
    this.t = t;
    this.pauseSec = Math.max(0.5, t.pause_threshold_sec);
    this.english = new Map(vocab.map((e) => [e.label, e.english_translation]));
    this.hindi = new Map(vocab.map((e) => [e.label, e.hindi_translation || e.english_translation]));
  }

  reset() {
    this.active = [];
    this.lastEventTime = null;
  }

  processEvent(e: SignEvent): Translation {
    if (e.confidence < this.t.min_confidence) {
      return {
        text: 'Uncertain sign', hindi_text: 'अनिश्चित संकेत', confidence: e.confidence, tokens: [e.label],
        source_signs: [e.label], timestamp: e.endTime, status: 'LOW_CONFIDENCE', is_final: false,
      };
    }
    this.active.push(e);
    this.lastEventTime = e.endTime;
    if (this.active.length >= Math.max(1, this.t.max_phrase_signs)) return this.finalize();
    return this.translateEvents(false);
  }

  checkPause(now: number): Translation | null {
    if (!this.active.length || this.lastEventTime === null) return null;
    return now - this.lastEventTime >= this.pauseSec ? this.finalize() : null;
  }

  private finalize(): Translation {
    const res = this.translateEvents(true);
    this.active = [];
    this.lastEventTime = null;
    return res;
  }

  private translateEvents(isFinal: boolean): Translation {
    const labels = this.active.map((e) => e.label);
    const tokens = labels.map((s) => s.toLowerCase().replace(/[_-]/g, ''));
    const template = PHRASE_TEMPLATES[tokens.join(' ')];
    const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();
    const text = template ? template[0] : tokens.map((s) => this.english.get(s) ?? capitalize(s)).join(' ') + '.';
    const hindi = template ? template[1] : tokens.map((s) => this.hindi.get(s) ?? s).join(' ') + '।';
    const conf = this.active.reduce((a, e) => a + e.confidence, 0) / this.active.length;
    return {
      text, hindi_text: hindi, confidence: Math.round(conf * 1e4) / 1e4, tokens, source_signs: labels,
      timestamp: this.active[this.active.length - 1].endTime, status: 'SUCCESS', is_final: isFinal,
    };
  }
}

// --------------------------------------------------------------------------------------------
// Pipeline (realtime_pipeline.py, segmentation mode)
// --------------------------------------------------------------------------------------------

export interface StepResult {
  frame: LandmarkFrame;
  /** A finished sign, ready for the classifier; pass its probabilities to acceptPrediction(). */
  segment: { frames: LandmarkFrame[]; input: ModelInput } | null;
}

export class SignPipeline {
  readonly settings: PipelineSettings;
  readonly vocab: VocabEntry[];
  private segmenter: SignSegmenter;
  private translator: Translator;
  private smoothedCenter: number[] | null = null;
  private smoothedScale: number | null = null;
  private signSequence: string[] = [];
  private segmentDisplayUntil = 0;
  private lastFrameTime = 0;
  fps = 0;
  latestPrediction: Prediction | null = null;
  latestLabel: string | null = null;
  latestTranslation: Translation | null = null;
  private pendingFinal: Translation | null = null;

  constructor(settings: PipelineSettings, vocab: VocabEntry[]) {
    this.settings = settings;
    this.vocab = vocab;
    this.segmenter = new SignSegmenter(settings.segmentation);
    this.translator = new Translator(settings.translation, vocab);
  }

  get signing() {
    return this.segmenter.inSegment;
  }

  get sequence() {
    return [...this.signSequence];
  }

  reset() {
    this.segmenter.reset();
    this.translator.reset();
    this.smoothedCenter = null;
    this.smoothedScale = null;
    this.signSequence = [];
    this.segmentDisplayUntil = 0;
    this.lastFrameTime = 0;
    this.latestPrediction = null;
    this.latestLabel = null;
    this.latestTranslation = null;
    this.pendingFinal = null;
  }

  /** Builds the LandmarkFrame for one extractor result (landmark_stream.process_frame). */
  makeFrame(raw: RawLandmarks, timestamp: number, width: number, height: number): LandmarkFrame {
    const n = this.settings.normalization;
    const cur = frameAnchor(raw.coords, raw.mask, n.min_scale_epsilon);
    let center = cur.center, scale = cur.scale;
    if (n.use_temporal_anchor_smoothing && this.smoothedCenter && this.smoothedScale !== null) {
      const a = n.anchor_smoothing_alpha;
      center = this.smoothedCenter.map((c, k) => (1 - a) * c + a * cur.center[k]);
      scale = (1 - a) * this.smoothedScale + a * cur.scale;
    }
    this.smoothedCenter = center;
    this.smoothedScale = scale;
    const normalized = new Float32Array(NODES * 3);
    applyAnchor(raw.coords, raw.mask, center, scale, n.z_scale_factor, normalized);

    // QualityChecker.compute_frame_quality + the GOOD/ACCEPTABLE gate (face mesh is off)
    const w = this.settings.quality.weights;
    const pose = raw.poseDetected ? raw.poseConfidence : 0;
    const lh = raw.leftHandDetected ? raw.leftHandConfidence : 0;
    const rh = raw.rightHandDetected ? raw.rightHandConfidence : 0;
    const q = Math.min(1, Math.max(0, w.pose * pose + w.dominant_hand * Math.max(lh, rh) + w.non_dominant_hand * Math.min(lh, rh)));
    const isValid = raw.poseDetected && (raw.leftHandDetected || raw.rightHandDetected) && q >= this.settings.quality.acceptable;

    const ref = n.reference_aspect || 0;
    const aspectYScale = ref > 0 && width > 0 && height > 0 ? ref / (width / height) : 1;
    return { timestamp, raw, normalized, aspectYScale, qualityScore: q, isValid };
  }

  /** Adds one frame. Returns a finished sign's model input when the hands have just dropped. */
  step(frame: LandmarkFrame): StepResult {
    if (this.lastFrameTime > 0) {
      const dt = frame.timestamp - this.lastFrameTime;
      if (dt > 0) this.fps = 0.85 * this.fps + 0.15 * (1 / dt);
    }
    this.lastFrameTime = frame.timestamp;

    const wasSigning = this.segmenter.inSegment;
    const seg = this.segmenter.update(frame);
    if (this.segmenter.inSegment && !wasSigning) this.latestLabel = null; // a new sign started
    if (!seg) return { frame, segment: null };
    const window = resampleByTime(seg, seg[0].timestamp, seg[seg.length - 1].timestamp, this.settings.sequence_length);
    return { frame, segment: { frames: seg, input: windowToInput(window, this.settings) } };
  }

  /** Classifier result for a segment from step(): accept or reject it, feed the translator. */
  acceptPrediction(probs: Float32Array, segment: LandmarkFrame[], now: number): Prediction {
    const order = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]);
    const cid = order[0], conf = probs[cid];
    const margin = order.length > 1 ? conf - probs[order[1]] : conf;
    const s = this.settings.segmentation;
    const accepted = conf >= s.min_confidence && margin >= s.min_margin;
    const v = this.vocab[cid];
    const quality = segment.reduce((a, f) => a + f.qualityScore, 0) / segment.length;
    this.latestPrediction = {
      class_id: cid, label: v.label, gloss: v.gloss, confidence: conf, input_quality: quality, accepted,
      top_k: order.slice(0, this.settings.top_k).map((i) => [i, this.vocab[i].label, probs[i]] as [number, string, number]),
    };
    if (accepted) {
      this.latestLabel = v.label;
      this.segmentDisplayUntil = now + 1.5;
      this.signSequence.push(v.label);
      if (this.signSequence.length > this.settings.max_events) this.signSequence.shift();
      const res = this.translator.processEvent({
        label: v.label, confidence: conf, startTime: segment[0].timestamp, endTime: segment[segment.length - 1].timestamp,
      });
      this.latestTranslation = res;
      if (res.is_final && res.text) {
        this.pendingFinal = res;
        this.latestTranslation = { ...res, is_final: false };
      }
    }
    return this.latestPrediction;
  }

  /** Translation to report for this frame: a phrase finalized now (is_final once), else the latest. */
  translationFor(now: number): Translation | null {
    if (this.pendingFinal) {
      const t = this.pendingFinal;
      this.pendingFinal = null;
      return t;
    }
    const pause = this.translator.checkPause(now);
    if (pause && pause.text) {
      this.latestTranslation = { ...pause, is_final: false };
      return pause;
    }
    return this.latestTranslation;
  }

  displayState(now: number): 'IDLE' | 'ACTIVE' | 'ENDING' {
    if (this.segmenter.inSegment) return 'ACTIVE';
    return now < this.segmentDisplayUntil ? 'ENDING' : 'IDLE';
  }
}
