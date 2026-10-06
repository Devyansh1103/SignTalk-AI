import React from 'react';
import type { VocabularyItem } from '../../types/realtime';
import { X, BookOpen, CheckCircle } from 'lucide-react';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  vocabulary: VocabularyItem[];
}

export const VocabularyModal: React.FC<Props> = ({ isOpen, onClose, vocabulary }) => {
  if (!isOpen) return null;

  return (
    <div
      id="vocabulary-modal"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-fade-in"
      role="dialog"
      aria-modal="true"
      aria-labelledby="vocab-modal-title"
    >
      <div className="w-full max-w-lg rounded-2xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
        {/* Modal Header */}
        <div className="flex items-center justify-between p-5 border-b border-slate-100 dark:border-slate-700">
          <div className="flex items-center gap-2">
            <BookOpen className="w-5 h-5 text-teal-600 dark:text-teal-400" />
            <h2 id="vocab-modal-title" className="text-base font-bold text-slate-900 dark:text-white">
              Supported Indian Sign Language Vocabulary ({vocabulary.length})
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
            aria-label="Close vocabulary dialog"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Content */}
        <div className="p-5 overflow-y-auto flex flex-col gap-3">
          <p className="text-xs text-slate-500 dark:text-slate-400">
            SignTalk AI currently recognizes the certified 10-gesture vocabulary (MVP-10).
            Execute gestures clearly with your upper body and hands within frame.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 pt-2">
            {vocabulary.map((item) => (
              <div
                key={item.class_id}
                className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900/60 border border-slate-200/60 dark:border-slate-700/60 flex items-center justify-between"
              >
                <div className="flex flex-col">
                  <span className="text-xs font-mono font-bold text-teal-700 dark:text-teal-300">
                    {item.gloss}
                  </span>
                  <span className="text-sm font-semibold text-slate-900 dark:text-white">
                    {item.english_translation}
                  </span>
                </div>
                <span className="text-sm font-medium text-slate-500 dark:text-slate-400">
                  {item.hindi_translation}
                </span>
              </div>
            ))}
          </div>

          <div className="p-3.5 rounded-xl bg-teal-50 dark:bg-teal-950/40 border border-teal-200 dark:border-teal-800/40 text-xs text-teal-800 dark:text-teal-200 flex items-start gap-2.5 mt-2">
            <CheckCircle className="w-4 h-4 shrink-0 text-teal-600 dark:text-teal-400 mt-0.5" />
            <span>
              Multi-sign combinations (e.g. &quot;HELLO TEACHER&quot; or &quot;GOOD MONDAY&quot;) are automatically composed into fluent English and Hindi sentences.
            </span>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="p-4 border-t border-slate-100 dark:border-slate-700 bg-slate-50 dark:bg-slate-900/30 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-teal-600 hover:bg-teal-500 text-white text-xs font-medium transition-colors"
          >
            Got it
          </button>
        </div>
      </div>
    </div>
  );
};
