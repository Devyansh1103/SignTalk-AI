/**
 * SignTalk AI - gloss -> sentence Transformer on onnxruntime-web. The model
 * (public/models/gloss_translator.onnx, written by scripts/export_gloss_onnx.py) takes the source
 * glosses with a language tag ("<2hi> i mother sick") plus the sentence so far, both padded with
 * <pad> (id 0) to a fixed length, and returns next-word probabilities for every target position;
 * translate() decodes greedily.
 *
 * A sentence is only used when every decoding step was confident (gloss_translator.json
 * min_confidence) and the model did not answer <unk>. Sign sequences without a learned pattern
 * come back as null, and the caller keeps the plain word-by-word text.
 */
import * as ort from 'onnxruntime-web';

interface SentenceMeta {
  src: string[];
  tgt: string[];
  bos: number;
  eos: number;
  /** The model's answer for sign sequences it learned to reject */
  unk: number;
  lang_tokens: { en: string; hi: string };
  /** Padded target / source lengths of the exported model */
  max_len: number;
  max_src: number;
  min_confidence: number;
}

const PUNCT = '.,!?।';

export class SentenceTranslator {
  private session: ort.InferenceSession;
  private meta: SentenceMeta;
  private srcIndex: Map<string, number>;

  private constructor(session: ort.InferenceSession, meta: SentenceMeta) {
    this.session = session;
    this.meta = meta;
    this.srcIndex = new Map(meta.src.map((t, i) => [t, i]));
  }

  static async create(): Promise<SentenceTranslator> {
    const base = `${import.meta.env.BASE_URL}models`;
    const [meta, session] = await Promise.all([
      fetch(`${base}/gloss_translator.json`).then((r) => r.json() as Promise<SentenceMeta>),
      ort.InferenceSession.create(`${base}/gloss_translator.onnx`, { executionProviders: ['wasm'] }),
    ]);
    return new SentenceTranslator(session, meta);
  }

  /** English or Hindi sentence for the glosses, or null when the model is not sure. */
  async translate(glosses: string[], lang: 'en' | 'hi'): Promise<string | null> {
    const ids = [this.meta.lang_tokens[lang], ...glosses].map((g) => this.srcIndex.get(g));
    const { max_len: T, max_src: S, tgt: vocab } = this.meta;
    if (glosses.length === 0 || ids.length > S || ids.some((i) => i === undefined)) return null;
    const srcIds = new BigInt64Array(S);
    ids.forEach((id, k) => (srcIds[k] = BigInt(id as number)));
    const src = new ort.Tensor('int64', srcIds, [1, S]);
    const out: number[] = [this.meta.bos];
    while (out.length < T) {
      const tgtIds = new BigInt64Array(T);
      out.forEach((id, k) => (tgtIds[k] = BigInt(id)));
      const all = (await this.session.run({ src, tgt: new ort.Tensor('int64', tgtIds, [1, T]) })).probs.data as Float32Array;
      const probs = all.subarray((out.length - 1) * vocab.length, out.length * vocab.length);
      let best = 0;
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
      if (probs[best] < this.meta.min_confidence || best === this.meta.unk) return null;
      if (best === this.meta.eos) return detokenize(out.slice(1).map((i) => this.meta.tgt[i]));
      out.push(best);
    }
    return null; // never finished the sentence
  }
}

function detokenize(tokens: string[]): string {
  return tokens.reduce((s, t) => (PUNCT.includes(t) || !s ? s + t : `${s} ${t}`), '');
}
