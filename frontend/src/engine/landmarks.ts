/**
 * SignTalk AI - MediaPipe Tasks (JS) landmark extractor, same models and settings as
 * src/preprocessing/landmark_extractor.py: pose_landmarker_full + hand_landmarker, IMAGE mode,
 * face mesh off. Output uses the 93-node layout (0-20 left hand, 21-41 right hand, 42-52 pose).
 */
import { FilesetResolver, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';
import { NODES, type RawLandmarks } from './pipeline';

// Same version as the npm package, so the WASM runtime matches the JS API
const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_BASE = `${import.meta.env.BASE_URL}models`;

const POSE_KEYPOINT_MAP = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16];

export interface ExtractorOptions {
  poseConfidence: number;
  handConfidence: number;
  handTrackingConfidence: number;
  delegate: 'CPU' | 'GPU';
}

export class LandmarkExtractor {
  private pose: PoseLandmarker;
  private hand: HandLandmarker;
  readonly delegate: string;

  private constructor(pose: PoseLandmarker, hand: HandLandmarker, delegate: string) {
    this.pose = pose;
    this.hand = hand;
    this.delegate = delegate;
  }

  static async create(o: ExtractorOptions): Promise<LandmarkExtractor> {
    const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
    const build = async (delegate: 'CPU' | 'GPU') => {
      const pose = await PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: `${MODEL_BASE}/pose_landmarker_full.task`, delegate },
        runningMode: 'IMAGE',
        numPoses: 1,
        minPoseDetectionConfidence: o.poseConfidence,
        minPosePresenceConfidence: o.poseConfidence,
        minTrackingConfidence: o.poseConfidence,
      });
      const hand = await HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: `${MODEL_BASE}/hand_landmarker.task`, delegate },
        runningMode: 'IMAGE',
        numHands: 2,
        minHandDetectionConfidence: o.handConfidence,
        minHandPresenceConfidence: o.handConfidence,
        minTrackingConfidence: o.handTrackingConfidence,
      });
      return new LandmarkExtractor(pose, hand, delegate);
    };
    if (o.delegate === 'GPU') {
      try {
        return await build('GPU');
      } catch (e) {
        console.warn('MediaPipe GPU delegate unavailable, using CPU:', e);
      }
    }
    return build('CPU');
  }

  extract(image: HTMLCanvasElement | HTMLVideoElement | ImageBitmap): RawLandmarks {
    const p = this.pose.detect(image);
    const h = this.hand.detect(image);
    const coords = new Float32Array(NODES * 3);
    const mask = new Uint8Array(NODES);
    const visibility = new Float32Array(NODES);
    const out: RawLandmarks = {
      coords, mask, visibility,
      poseDetected: false, leftHandDetected: false, rightHandDetected: false,
      poseConfidence: 0, leftHandConfidence: 0, rightHandConfidence: 0,
    };

    (h.landmarks ?? []).forEach((lms, k) => {
      const cat = h.handedness[k]?.[0];
      if (!cat) return;
      const left = cat.categoryName === 'Left';
      const off = left ? 0 : 21;
      if (left) {
        out.leftHandDetected = true;
        out.leftHandConfidence = cat.score;
      } else {
        out.rightHandDetected = true;
        out.rightHandConfidence = cat.score;
      }
      lms.slice(0, 21).forEach((lm, j) => {
        const q = off + j;
        coords[q * 3] = lm.x;
        coords[q * 3 + 1] = lm.y;
        coords[q * 3 + 2] = lm.z;
        visibility[q] = cat.score;
        mask[q] = 1;
      });
    });

    const pose = p.landmarks?.[0];
    if (pose) {
      out.poseDetected = true;
      let sum = 0;
      POSE_KEYPOINT_MAP.forEach((pi, k) => {
        const lm = pose[pi], q = 42 + k;
        if (!lm) return;
        coords[q * 3] = lm.x;
        coords[q * 3 + 1] = lm.y;
        coords[q * 3 + 2] = lm.z;
        visibility[q] = lm.visibility ?? 1;
        mask[q] = 1;
        sum += visibility[q];
      });
      out.poseConfidence = sum / POSE_KEYPOINT_MAP.length;
    }
    return out;
  }

  close() {
    this.pose.close();
    this.hand.close();
  }
}
