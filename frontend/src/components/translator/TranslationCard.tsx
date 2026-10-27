import { Fragment, useState } from 'react';
import { ArrowRight, Check, Copy, RotateCcw } from 'lucide-react';
import type { TranslationPayload, VocabularyItem } from '../../types/realtime';
import { Button } from '../ui/Button';

interface Props {
  translation?: TranslationPayload;
  sequence: string[];
  vocab: Map<string, VocabularyItem>;
  largeText: boolean;
  onClear: () => void;
}

export function TranslationCard({ translation, sequence, vocab, largeText, onClear }: Props) {
  const [copied, setCopied] = useState(false);
  const english = translation?.text || '';
  const hindi = translation?.hindi_text || '';

  const copy = () => {
    if (!english) return;
    navigator.clipboard.writeText(hindi ? `${english} (${hindi})` : english).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <section aria-labelledby="translation-title" className="card p-5">
      <h2 id="translation-title" className="text-sm font-medium text-ink-2 mb-3">
        Translated Sentence
      </h2>

      <div className="rounded-xl bg-surface-2 border border-line px-4 py-3 min-h-[4.25rem]" aria-live="polite">
        {english ? (
          <>
            <p className={`font-semibold text-ink ${largeText ? 'text-2xl' : 'text-xl'}`}>{english}</p>
            {hindi && <p className={`text-accent-text mt-0.5 ${largeText ? 'text-lg' : 'text-base'}`}>{hindi}</p>}
          </>
        ) : (
          <p className="text-sm text-ink-3 py-2">Sign one or more words; the sentence appears here.</p>
        )}
      </div>

      <div className="mt-4">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-ink-3 mb-1.5">Live gesture sequence</p>
        <div className="flex items-center gap-2 overflow-x-auto no-scrollbar py-0.5 text-xs">
          {sequence.length ? (
            sequence.map((s, i) => (
              <Fragment key={i}>
                <span className="px-2.5 py-1 rounded-lg bg-accent-soft text-accent-text font-medium shrink-0">
                  {vocab.get(s)?.english_translation ?? s}
                </span>
                {i < sequence.length - 1 && <ArrowRight className="w-3.5 h-3.5 text-ink-3 shrink-0" aria-hidden="true" />}
              </Fragment>
            ))
          ) : (
            <span className="text-ink-3">No signs yet</span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 mt-4">
        <Button variant="secondary" onClick={onClear}>
          <RotateCcw className="w-4 h-4" />
          Clear
        </Button>
        <Button variant="primary" onClick={copy} disabled={!english} aria-label="Copy translated text">
          {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
          {copied ? 'Copied' : 'Copy'}
        </Button>
      </div>
    </section>
  );
}
