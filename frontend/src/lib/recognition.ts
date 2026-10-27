import type { ServerRealtimeMessage } from '../types/realtime';

export type SignState = 'waiting' | 'signing' | 'recognized' | 'uncertain';

export interface SignView {
  state: SignState;
  /** Vocabulary label of the recognised sign (only when state is 'recognized') */
  label?: string;
  confidence?: number;
  /** True while the result of the sign that just ended is on screen (vs. an older sign kept for reference) */
  fresh: boolean;
}

/**
 * What the sign card should show, from the pipeline's per-frame message.
 * smoothed_label is only set for an accepted sign, so a rejected guess is never shown as the result.
 */
export function describeSign(msg: ServerRealtimeMessage | null): SignView {
  if (!msg) return { state: 'waiting', fresh: false };
  const s = msg.state_machine;
  if (s === 'ACTIVE' || s === 'CANDIDATE') return { state: 'signing', fresh: false };

  const p = msg.prediction;
  const accepted = !!msg.smoothed_label && !!p && p.label === msg.smoothed_label;
  if (accepted) return { state: 'recognized', label: msg.smoothed_label!, confidence: p!.confidence, fresh: s === 'ENDING' };
  // The last sign was classified but rejected (too unsure); stays until the next sign starts
  if (p) return { state: 'uncertain', fresh: true };
  return { state: 'waiting', fresh: false };
}
