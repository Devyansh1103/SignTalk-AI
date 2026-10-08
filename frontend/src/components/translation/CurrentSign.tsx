import React from 'react';
import type { PredictionPayload } from '../../types/realtime';
import { ConfidenceIndicator } from './ConfidenceIndicator';
import { Sparkles, Activity } from 'lucide-react';

interface Props {
  prediction?: PredictionPayload;
  smoothedLabel?: string | null;
  stateMachine: string;
  isLargeText?: boolean;
}

export const CurrentSign: React.FC<Props> = ({
  prediction,
  smoothedLabel,
  stateMachine,
  isLargeText,
}) => {
  const isIdle = !stateMachine || stateMachine === 'IDLE';
  const isCandidate = stateMachine === 'CANDIDATE';
  const isActive = stateMachine === 'ACTIVE';
  const isEnding = stateMachine === 'ENDING';

  // Confidence check against certified 0.65 threshold
  const hasConfidentPrediction = !!prediction && prediction.confidence >= 0.65;
  const isUncertain = !hasConfidentPrediction || smoothedLabel === 'UNCERTAIN';

  // Determine display sign based strictly on lifecycle state
  let displaySign = '---';
  let subtitle = 'Ready for sign gesture';

  if (isIdle) {
    displaySign = '---';
    subtitle = 'Position hands in camera view to sign';
  } else if (isCandidate) {
    if (hasConfidentPrediction && prediction?.label) {
      displaySign = prediction.label.toUpperCase();
      subtitle = 'Detecting gesture...';
    } else {
      displaySign = '---';
      subtitle = 'Detecting gesture...';
    }
  } else if (isActive) {
    if (smoothedLabel && smoothedLabel !== 'UNCERTAIN' && smoothedLabel !== 'UNSTABLE') {
      displaySign = smoothedLabel.toUpperCase();
    } else if (hasConfidentPrediction && prediction?.label) {
      displaySign = prediction.label.toUpperCase();
    }
    subtitle = 'Active sign recognized';
  } else if (isEnding) {
    if (smoothedLabel && smoothedLabel !== 'UNCERTAIN' && smoothedLabel !== 'UNSTABLE') {
      displaySign = smoothedLabel.toUpperCase();
    } else if (prediction?.label) {
      displaySign = prediction.label.toUpperCase();
    }
    subtitle = 'Gesture concluded';
  }

  const getStateBadge = () => {
    switch (stateMachine) {
      case 'ACTIVE':
        return { label: 'Active Gesture', color: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/30' };
      case 'CANDIDATE':
        return { label: 'Detecting...', color: 'bg-teal-500/10 text-teal-600 dark:text-teal-400 border-teal-500/30' };
      case 'ENDING':
        return { label: 'Concluding', color: 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/30' };
      default:
        return { label: 'Idle / Ready', color: 'bg-slate-500/10 text-slate-500 dark:text-slate-400 border-slate-500/30' };
    }
  };

  const stateBadge = getStateBadge();

  return (
    <div
      id="current-sign-panel"
      className="p-5 rounded-2xl bg-white dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700/80 shadow-sm flex flex-col justify-between gap-4 transition-all"
    >
      {/* Header with state indicator */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
          <Sparkles className="w-4 h-4 text-teal-600 dark:text-teal-400" />
          <span>Recognized Sign</span>
        </div>
        <div className={`px-2.5 py-0.5 rounded-full text-[11px] font-medium border ${stateBadge.color}`}>
          {stateBadge.label}
        </div>
      </div>

      {/* Main Sign Display */}
      <div className="py-2 flex flex-col items-start justify-center">
        <div
          id="recognized-sign-display"
          className={`font-black tracking-tight transition-all ${
            isIdle
              ? 'text-slate-300 dark:text-slate-600'
              : isCandidate
              ? 'text-teal-600 dark:text-teal-400 animate-pulse'
              : 'text-slate-900 dark:text-white'
          } ${isLargeText ? 'text-5xl lg:text-6xl' : 'text-4xl lg:text-5xl'}`}
          aria-live="polite"
        >
          {displaySign}
        </div>

        <div className="flex items-center gap-2 mt-1.5 text-xs text-slate-500 dark:text-slate-400">
          <span>{subtitle}</span>
          {!isIdle && prediction && prediction.input_quality > 0 && (
            <>
              <span>•</span>
              <div className="flex items-center gap-1">
                <Activity className="w-3 h-3" />
                <span>Quality: {(prediction.input_quality * 100).toFixed(0)}%</span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Confidence Component */}
      <ConfidenceIndicator
        confidence={!isIdle && prediction ? prediction.confidence : 0}
        isUncertain={isUncertain}
        isIdle={isIdle}
      />

      {/* Top Alternative Guesses (only shown during active recognition) */}
      {!isIdle && prediction && prediction.top_k && prediction.top_k.length > 1 && (
        <div className="pt-2 border-t border-slate-100 dark:border-slate-700/50 flex items-center gap-3 text-xs text-slate-500 dark:text-slate-400">
          <span className="font-medium text-[11px] uppercase tracking-wide">Alternatives:</span>
          {prediction.top_k.slice(1, 3).map(([cid, lbl, conf]) => (
            <span key={cid} className="font-mono">
              {lbl} ({(conf * 100).toFixed(0)}%)
            </span>
          ))}
        </div>
      )}
    </div>
  );
};
