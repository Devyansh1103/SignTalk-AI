import { History, Trash2 } from 'lucide-react';
import type { FinalizedPhrase } from '../../types/realtime';

interface Props {
  transcript: FinalizedPhrase[];
  onClear: () => void;
  largeText: boolean;
}

export function TranscriptCard({ transcript, onClear, largeText }: Props) {
  return (
    <section aria-labelledby="transcript-title" className="card p-5">
      <div className="flex items-center justify-between mb-3">
        <h2 id="transcript-title" className="flex items-center gap-2 text-sm font-medium text-ink-2">
          <History className="w-4 h-4 text-accent-text" aria-hidden="true" />
          Session Transcript ({transcript.length})
        </h2>
        {transcript.length > 0 && (
          <button type="button" onClick={onClear} className="inline-flex items-center gap-1.5 text-xs text-ink-3 hover:text-bad">
            <Trash2 className="w-3.5 h-3.5" aria-hidden="true" />
            Clear
          </button>
        )}
      </div>
      <div role="log" aria-live="polite" className="flex flex-col gap-2 max-h-60 overflow-y-auto pr-1">
        {transcript.length ? (
          transcript.map((t, i) => (
            <article key={t.id} className="rounded-xl bg-surface-2 border border-line px-3 py-2.5">
              <div className="flex items-center justify-between text-[11px] text-ink-3 mb-0.5">
                <span>Sentence {i + 1}</span>
                <span className="font-mono">{Math.round(t.confidence * 100)}%</span>
              </div>
              <p className={`font-medium text-ink ${largeText ? 'text-lg' : 'text-sm'}`}>{t.english}</p>
              {t.hindi && <p className={`text-accent-text ${largeText ? 'text-base' : 'text-sm'}`}>{t.hindi}</p>}
            </article>
          ))
        ) : (
          <p className="text-sm text-ink-3 py-4 text-center">Finished sentences collect here when you pause between signs.</p>
        )}
      </div>
    </section>
  );
}
