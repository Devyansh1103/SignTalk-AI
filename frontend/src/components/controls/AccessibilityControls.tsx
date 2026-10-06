import React from 'react';
import type { AccessibilitySettings } from '../../types/realtime';
import { Type, Eye, FlipHorizontal, Gauge, BookOpen, Shield } from 'lucide-react';

interface Props {
  settings: AccessibilitySettings;
  onUpdateSettings: (updater: (prev: AccessibilitySettings) => AccessibilitySettings) => void;
  onOpenVocabulary: () => void;
  onOpenPrivacy: () => void;
}

export const AccessibilityControls: React.FC<Props> = ({
  settings,
  onUpdateSettings,
  onOpenVocabulary,
  onOpenPrivacy,
}) => {
  return (
    <div
      id="accessibility-controls-bar"
      className="flex flex-wrap items-center justify-between gap-2 p-3 rounded-2xl bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 text-xs text-slate-600 dark:text-slate-300"
      aria-label="Accessibility and display settings"
    >
      <div className="flex flex-wrap items-center gap-1.5">
        {/* Large Text Toggle */}
        <button
          id="toggle-large-text-button"
          onClick={() =>
            onUpdateSettings((s) => ({ ...s, largeText: !s.largeText }))
          }
          className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border font-medium transition-colors ${
            settings.largeText
              ? 'bg-teal-600 text-white border-teal-600'
              : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 hover:bg-slate-100'
          }`}
          aria-pressed={settings.largeText}
        >
          <Type className="w-3.5 h-3.5" />
          <span>Large Text</span>
        </button>

        {/* High Contrast Toggle */}
        <button
          id="toggle-high-contrast-button"
          onClick={() =>
            onUpdateSettings((s) => ({ ...s, highContrast: !s.highContrast }))
          }
          className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border font-medium transition-colors ${
            settings.highContrast
              ? 'bg-slate-900 text-white border-slate-900 dark:bg-white dark:text-slate-900'
              : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 hover:bg-slate-100'
          }`}
          aria-pressed={settings.highContrast}
        >
          <Eye className="w-3.5 h-3.5" />
          <span>High Contrast</span>
        </button>

        {/* Mirror Preview Toggle */}
        <button
          id="toggle-mirror-preview-button"
          onClick={() =>
            onUpdateSettings((s) => ({ ...s, mirrorPreview: !s.mirrorPreview }))
          }
          className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border font-medium transition-colors ${
            settings.mirrorPreview
              ? 'bg-teal-600 text-white border-teal-600'
              : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 hover:bg-slate-100'
          }`}
          aria-pressed={settings.mirrorPreview}
        >
          <FlipHorizontal className="w-3.5 h-3.5" />
          <span>Mirror Video</span>
        </button>

        {/* Debug Metrics Toggle */}
        <button
          id="toggle-debug-metrics-button"
          onClick={() =>
            onUpdateSettings((s) => ({ ...s, showDebugMetrics: !s.showDebugMetrics }))
          }
          className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border font-medium transition-colors ${
            settings.showDebugMetrics
              ? 'bg-slate-800 text-teal-300 border-slate-700'
              : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 hover:bg-slate-100'
          }`}
          aria-pressed={settings.showDebugMetrics}
        >
          <Gauge className="w-3.5 h-3.5" />
          <span>Telemetry</span>
        </button>
      </div>

      <div className="flex items-center gap-1.5">
        {/* Supported Vocabulary Modal Button */}
        <button
          id="open-vocabulary-modal-button"
          onClick={onOpenVocabulary}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 font-medium hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
        >
          <BookOpen className="w-3.5 h-3.5 text-teal-600 dark:text-teal-400" />
          <span>Supported Signs</span>
        </button>

        {/* Privacy Notice Button */}
        <button
          id="open-privacy-modal-button"
          onClick={onOpenPrivacy}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 font-medium hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
        >
          <Shield className="w-3.5 h-3.5 text-slate-500" />
          <span>Privacy</span>
        </button>
      </div>
    </div>
  );
};
