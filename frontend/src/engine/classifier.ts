/**
 * SignTalk AI - ST-GCN classifier on onnxruntime-web. The model (public/models/stgcn.onnx,
 * written by scripts/export_onnx.py) takes x [1,3,45,93] + mask [1,1,45,93] and returns softmax
 * probabilities [1,141].
 */
import * as ort from 'onnxruntime-web';
import type { ModelInput } from './pipeline';

// Vite bundles the WASM runtime next to the app. Single-threaded: threads need cross-origin
// isolation headers, and one forward pass per sign is fast enough without them.
ort.env.wasm.numThreads = 1;

export class SignClassifier {
  private session: ort.InferenceSession;
  private T: number;

  private constructor(session: ort.InferenceSession, T: number) {
    this.session = session;
    this.T = T;
  }

  static async create(sequenceLength: number): Promise<SignClassifier> {
    const session = await ort.InferenceSession.create(`${import.meta.env.BASE_URL}models/stgcn.onnx`, {
      executionProviders: ['wasm'],
    });
    const c = new SignClassifier(session, sequenceLength);
    await c.classify({ x: new Float32Array(3 * sequenceLength * 93), mask: new Float32Array(sequenceLength * 93) });
    return c;
  }

  async classify(input: ModelInput): Promise<Float32Array> {
    const out = await this.session.run({
      x: new ort.Tensor('float32', input.x, [1, 3, this.T, 93]),
      mask: new ort.Tensor('float32', input.mask, [1, 1, this.T, 93]),
    });
    return out.probs.data as Float32Array;
  }
}
