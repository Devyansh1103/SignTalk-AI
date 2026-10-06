import React from 'react';
import { Shield, BookOpen } from 'lucide-react';

interface Props {
  onOpenVocabulary: () => void;
  onOpenPrivacy: () => void;
}

export const Footer: React.FC<Props> = ({ onOpenVocabulary, onOpenPrivacy }) => {
  return (
    <footer className="w-full border-t border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 py-6 mt-12 transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500 dark:text-slate-400">
        <div className="flex flex-col sm:flex-row items-center gap-2 sm:gap-4 text-center sm:text-left">
          <span>SignTalk AI — GLA University Academic Research Project</span>
          <span className="hidden sm:inline">|</span>
          <span>Spatial-Temporal Graph & Transformer Architecture</span>
        </div>

        <div className="flex items-center gap-4">
          <button
            onClick={onOpenVocabulary}
            className="inline-flex items-center gap-1 hover:text-teal-600 dark:hover:text-teal-400 transition-colors"
          >
            <BookOpen className="w-3.5 h-3.5" />
            <span>Supported Vocabulary</span>
          </button>
          <button
            onClick={onOpenPrivacy}
            className="inline-flex items-center gap-1 hover:text-teal-600 dark:hover:text-teal-400 transition-colors"
          >
            <Shield className="w-3.5 h-3.5" />
            <span>Privacy Notice</span>
          </button>
        </div>
      </div>
    </footer>
  );
};
