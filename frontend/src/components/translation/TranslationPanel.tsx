import React, { useState } from 'react';
import type { TranslationPayload } from '../../types/realtime';
import { Languages, Copy, Check, ArrowRight } from 'lucide-react';

interface Props {
  translation?: TranslationPayload;
  signSequence: string[];
  isLargeText?: boolean;
}

export const TranslationPanel: React.FC<Props> = ({
  translation,
  signSequence,
  isLargeText,
}) => {
  const [copied, setCopied] = useState(false);

  const englishText = translation?.text || '';
  const hindiText = translation?.hindi_text || '';

  const handleCopy = () => {
    if (!englishText) return;
    const fullText = hindiText ? `${englishText} (${hindiText})` : englishText;
    navigator.clipboard.writeText(fullText).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div
      id="translation-panel-container"
      className="p-5 rounded-2xl bg-white dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700/80 shadow-sm flex flex-col justify-between gap-4 transition-all"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
          <Languages className="w-4 h-4 text-teal-600 dark:text-teal-400" />
          <span>Natural Language Translation</span>
        </div>
        {englishText && (
          <button
            id="copy-translation-button"
            onClick={handleCopy}
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
            aria-label="Copy translated text to clipboard"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-600" />
                <span className="text-emerald-600">Copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5" />
                <span>Copy</span>
              </>
            )}
          </button>
        )}
      </div>

      {/* Main Sentence Display */}
      <div className="py-2 flex flex-col gap-2">
        {englishText ? (
          <>
            <div
              id="english-translation-text"
              className={`font-bold text-slate-900 dark:text-white leading-tight ${
                isLargeText ? 'text-3xl lg:text-4xl' : 'text-2xl lg:text-3xl'
              }`}
              aria-live="polite"
            >
              &quot;{englishText}&quot;
            </div>
            {hindiText && (
              <div
                id="hindi-translation-text"
                className={`font-medium text-teal-700 dark:text-teal-300 ${
                  isLargeText ? 'text-xl' : 'text-lg'
                }`}
                aria-live="polite"
              >
                {hindiText}
              </div>
            )}
          </>
        ) : (
          <div className="text-sm text-slate-400 dark:text-slate-500 italic py-3">
            Start signing to generate translated natural language text...
          </div>
        )}
      </div>

      {/* Live Sign Sequence Stream */}
      <div className="pt-3 border-t border-slate-100 dark:border-slate-700/50 flex flex-col gap-1.5">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500">
          Live Gesture Sequence:
        </span>
        <div
          id="live-sequence-bar"
          className="flex items-center gap-2 overflow-x-auto py-1 text-xs text-slate-700 dark:text-slate-300 no-scrollbar"
        >
          {signSequence.length > 0 ? (
            signSequence.map((sign, idx) => (
              <React.Fragment key={idx}>
                <span className="px-2.5 py-1 rounded-lg bg-teal-50 dark:bg-teal-950/40 text-teal-700 dark:text-teal-300 font-mono font-medium border border-teal-200 dark:border-teal-800/40 shrink-0">
                  {sign.toUpperCase()}
                </span>
                {idx < signSequence.length - 1 && (
                  <ArrowRight className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                )}
              </React.Fragment>
            ))
          ) : (
            <span className="text-slate-400 dark:text-slate-500 italic text-xs">
              No signs in active sequence
            </span>
          )}
        </div>
      </div>
    </div>
  );
};
