import type { SignView } from '../../lib/recognition';
import type { VocabularyItem } from '../../types/realtime';

interface Props {
  view: SignView;
  vocab: Map<string, VocabularyItem>;
  cameraOn: boolean;
  largeText: boolean;
}

const BADGES = {
  waiting: { label: 'Waiting', tone: 'text-ink-3 border-line bg-surface-2' },
  signing: { label: 'Signing…', tone: 'text-accent-text border-accent/40 bg-accent-soft' },
  recognized: { label: 'Recognised', tone: 'text-ok border-ok/30 bg-ok/10' },
  uncertain: { label: 'Not sure', tone: 'text-warn border-warn/30 bg-warn/10' },
};

export function DetectedSign({ view, vocab, cameraOn, largeText }: Props) {
  const entry = view.label ? vocab.get(view.label) : undefined;
  const pct = view.confidence !== undefined ? Math.round(view.confidence * 100) : null;
  const badge = BADGES[view.state];

  let hint: string;
  if (!cameraOn) hint = 'Start the camera to begin.';
  else if (view.state === 'signing') hint = 'Finish the sign, then lower your hands.';
  else if (view.state === 'uncertain') hint = 'That sign was unclear. Try again a little slower, with your hands fully in view.';
  else if (view.state === 'recognized' && !view.fresh) hint = 'Last recognised sign. Raise your hands to sign the next word.';
  else if (view.state === 'recognized') hint = 'Sign recognised.';
  else hint = 'Raise your hands and sign a word.';

  return (
    <section aria-labelledby="detected-sign-title" className="card p-5">
      <div className="flex items-center justify-between gap-3 mb-3">
        <h2 id="detected-sign-title" className="text-sm font-medium text-ink-2">
          Detected Sign
        </h2>
        <span className={`px-2.5 py-0.5 rounded-full text-[11px] font-medium border ${badge.tone}`}>{badge.label}</span>
      </div>

      <div aria-live="polite" className={`min-h-[5.5rem] ${view.state === 'recognized' && !view.fresh ? 'opacity-70' : ''}`}>
        {view.state === 'recognized' ? (
          <>
            <p className={`font-bold tracking-tight text-ink leading-tight ${largeText ? 'text-5xl' : 'text-4xl'}`}>
              {entry?.english_translation ?? view.label}
            </p>
            {entry?.hindi_translation && <p className={`text-accent-text mt-1 ${largeText ? 'text-2xl' : 'text-xl'}`}>{entry.hindi_translation}</p>}
          </>
        ) : (
          <p className={`font-bold tracking-tight text-ink-3/60 ${largeText ? 'text-5xl' : 'text-4xl'}`} aria-label="No sign">
            —
          </p>
        )}
      </div>

      <div className="mt-3">
        <div className="flex items-center justify-between text-xs mb-1.5">
          <span className="text-ink-2">Confidence</span>
          <span className="font-mono font-semibold text-ink">{pct !== null ? `${pct}%` : '—'}</span>
        </div>
        <div
          className="h-2 rounded-full bg-surface-3 overflow-hidden"
          role="progressbar"
          aria-label="Confidence"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={pct ?? 0}
        >
          <div className="h-full rounded-full bg-gradient-accent transition-[width] duration-300" style={{ width: `${pct ?? 0}%` }} />
        </div>
      </div>
      <p className="text-xs text-ink-3 mt-3">{hint}</p>
    </section>
  );
}
