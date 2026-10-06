import React from 'react';
import type { ConfidenceTier } from '../../types/realtime';

interface Props {
  confidence: number; // 0.0 to 1.0
  isUncertain?: boolean;
}

export const ConfidenceIndicator: React.FC<Props> = ({ confidence, isUncertain }) => {
  const percentage = Math.round(confidence * 100);

  const getTier = (): { tier: ConfidenceTier; label: string; colorClass: string; barColor: string } => {
    if (isUncertain || confidence < 0.40) {
      return {
        tier: 'uncertain',
        label: 'Uncertain',
        colorClass: 'bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-500/30',
        barColor: 'bg-amber-500',
      };
    }
    if (confidence >= 0.65) {
      return {
        tier: 'high',
        label: 'High Confidence',
        colorClass: 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-500/30',
        barColor: 'bg-emerald-500',
      };
    }
    if (confidence >= 0.50) {
      return {
        tier: 'moderate',
        label: 'Moderate',
        colorClass: 'bg-teal-500/10 text-teal-700 dark:text-teal-400 border-teal-500/30',
        barColor: 'bg-teal-500',
      };
    }
    return {
      tier: 'low',
      label: 'Low',
      colorClass: 'bg-rose-500/10 text-rose-700 dark:text-rose-400 border-rose-500/30',
      barColor: 'bg-rose-500',
    };
  };

  const { label, colorClass, barColor } = getTier();

  return (
    <div id="confidence-indicator-container" className="flex flex-col gap-1.5 w-full">
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-500 dark:text-slate-400 font-medium">Confidence Score</span>
        <div className="flex items-center gap-2">
          <span className="font-mono font-semibold text-slate-800 dark:text-slate-200">
            {isUncertain ? '--%' : `${percentage}%`}
          </span>
          <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium border ${colorClass}`}>
            {label}
          </span>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
        <div
          className={`h-full transition-all duration-300 rounded-full ${barColor}`}
          style={{ width: `${isUncertain ? 0 : Math.min(100, percentage)}%` }}
          role="progressbar"
          aria-valuenow={percentage}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>
    </div>
  );
};
