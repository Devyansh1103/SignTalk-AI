import type { PredictionPayload, VocabularyItem } from '../../types/realtime';

/** The model's top guesses for the most recent sign (accepted or not). */
export function TopPredictions({ prediction, vocab }: { prediction?: PredictionPayload; vocab: Map<string, VocabularyItem> }) {
  const rows = prediction?.top_k?.slice(0, 3) ?? [];

  return (
    <section aria-labelledby="top-predictions-title" className="card p-5">
      <h2 id="top-predictions-title" className="text-sm font-medium text-ink-2 mb-3">
        Top Predictions
      </h2>
      {rows.length ? (
        <ol className="flex flex-col gap-2.5">
          {rows.map(([cid, label, conf], i) => (
            <li key={cid} className="grid grid-cols-[1.75rem_minmax(0,7rem)_1fr_3rem] items-center gap-3 text-sm">
              <span className="w-7 h-7 rounded-lg border border-line bg-surface-2 flex items-center justify-center text-xs font-semibold text-ink-2">
                {i + 1}
              </span>
              <span className="truncate text-ink">{vocab.get(label)?.english_translation ?? label}</span>
              <span className="h-1.5 rounded-full bg-surface-3 overflow-hidden" aria-hidden="true">
                <span className="block h-full rounded-full bg-gradient-accent" style={{ width: `${Math.round(conf * 100)}%` }} />
              </span>
              <span className="font-mono text-xs text-ink-2 text-right">{conf.toFixed(2)}</span>
            </li>
          ))}
        </ol>
      ) : (
        <p className="text-sm text-ink-3">The model's best guesses appear here after each sign.</p>
      )}
    </section>
  );
}
