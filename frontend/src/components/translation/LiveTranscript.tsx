import React from 'react';
import type { FinalizedPhrase } from '../../types/realtime';
import { History, Trash2 } from 'lucide-react';

interface Props {
  transcript: FinalizedPhrase[];
  onClear: () => void;
  isLargeText?: boolean;
}

export const LiveTranscript: React.FC<Props> = ({
  transcript,
  onClear,
  isLargeText,
}) => {
  return (
    <div
      id="live-transcript-container"
      className="p-5 rounded-2xl bg-white dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700/80 shadow-sm flex flex-col gap-3 transition-all"
    >
      {/* Header */}
      <div className="flex items-center justify-between pb-2 border-b border-slate-100 dark:border-slate-700/50">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
          <History className="w-4 h-4 text-teal-600 dark:text-teal-400" />
          <span>Session Transcript ({transcript.length})</span>
        </div>
        {transcript.length > 0 && (
          <button
            id="clear-transcript-button"
            onClick={onClear}
            className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-lg text-xs font-medium text-slate-500 hover:text-rose-600 dark:hover:text-rose-400 transition-colors"
            title="Clear transcript history"
            aria-label="Clear transcript history"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear</span>
          </button>
        )}
      </div>

      {/* Transcript Items List */}
      <div
        id="transcript-scroll-area"
        className="flex flex-col gap-3 max-h-56 overflow-y-auto pr-1 text-slate-800 dark:text-slate-200"
        role="log"
        aria-live="polite"
      >
        {transcript.length > 0 ? (
          transcript.map((item, idx) => (
            <div
              key={item.id || idx}
              className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900/50 border border-slate-100 dark:border-slate-800 flex flex-col gap-1 transition-all"
            >
              <div className="flex items-center justify-between text-[11px] text-slate-400 dark:text-slate-500">
                <span>Phrase #{idx + 1}</span>
                <span className="font-mono">{(item.confidence * 100).toFixed(0)}% Conf</span>
              </div>
              <div className={`font-semibold text-slate-900 dark:text-white ${isLargeText ? 'text-lg' : 'text-base'}`}>
                &quot;{item.english}&quot;
              </div>
              {item.hindi && (
                <div className={`text-teal-700 dark:text-teal-300 font-medium ${isLargeText ? 'text-base' : 'text-sm'}`}>
                  {item.hindi}
                </div>
              )}
            </div>
          ))
        ) : (
          <div className="text-center py-8 text-sm text-slate-400 dark:text-slate-500 italic">
            Finalized sentences will appear here as you pause between sign expressions.
          </div>
        )}
      </div>
    </div>
  );
};
