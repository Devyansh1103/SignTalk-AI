import React, { useEffect } from 'react';
import type { CameraStatus } from '../../types/realtime';
import { Play, Pause, RotateCcw, Square } from 'lucide-react';

interface Props {
  cameraStatus: CameraStatus;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onReset: () => void;
  onStop: () => void;
}

export const SessionControls: React.FC<Props> = ({
  cameraStatus,
  onStart,
  onPause,
  onResume,
  onReset,
  onStop,
}) => {
  const isRunning = cameraStatus === 'running';
  const isPaused = cameraStatus === 'paused';
  const isIdle = cameraStatus === 'idle' || cameraStatus === 'stopped';

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in an input
      if (['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) {
        return;
      }

      if (e.code === 'Space') {
        e.preventDefault();
        if (isRunning) onPause();
        else if (isPaused) onResume();
        else if (isIdle) onStart();
      } else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        onReset();
      } else if (e.key === 's' || e.key === 'S') {
        e.preventDefault();
        if (!isIdle) onStop();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isRunning, isPaused, isIdle, onPause, onResume, onStart, onReset, onStop]);

  return (
    <div
      id="session-controls-bar"
      className="flex flex-wrap items-center justify-between gap-3 p-4 rounded-2xl bg-white dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700/80 shadow-sm"
      role="toolbar"
      aria-label="Recognition Session Controls"
    >
      <div className="flex items-center gap-2">
        {/* Start / Resume */}
        {isIdle && (
          <button
            id="control-start-button"
            onClick={onStart}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-teal-600 hover:bg-teal-500 text-white font-medium shadow-md shadow-teal-900/10 transition-all focus:outline-none focus:ring-2 focus:ring-teal-400"
            aria-label="Start Camera and Real-Time Recognition (Shortcut: Space)"
          >
            <Play className="w-4 h-4 fill-current" />
            <span>Start (Space)</span>
          </button>
        )}

        {/* Pause */}
        {isRunning && (
          <button
            id="control-pause-button"
            onClick={onPause}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-amber-500 hover:bg-amber-400 text-slate-900 font-medium shadow-md shadow-amber-900/10 transition-all focus:outline-none focus:ring-2 focus:ring-amber-400"
            aria-label="Pause Recognition (Shortcut: Space)"
          >
            <Pause className="w-4 h-4 fill-current" />
            <span>Pause (Space)</span>
          </button>
        )}

        {/* Resume */}
        {isPaused && (
          <button
            id="control-resume-button"
            onClick={onResume}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-teal-600 hover:bg-teal-500 text-white font-medium shadow-md shadow-teal-900/10 transition-all focus:outline-none focus:ring-2 focus:ring-teal-400"
            aria-label="Resume Recognition (Shortcut: Space)"
          >
            <Play className="w-4 h-4 fill-current" />
            <span>Resume (Space)</span>
          </button>
        )}

        {/* Reset Session */}
        <button
          id="control-reset-button"
          onClick={onReset}
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-100 hover:bg-slate-200 dark:bg-slate-700 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-200 font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-slate-400"
          aria-label="Reset Sequence and Transcript (Shortcut: R)"
        >
          <RotateCcw className="w-4 h-4" />
          <span>Reset (R)</span>
        </button>
      </div>

      {/* Stop Camera */}
      {!isIdle && (
        <button
          id="control-stop-button"
          onClick={onStop}
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-rose-50 hover:bg-rose-100 dark:bg-rose-950/30 dark:hover:bg-rose-900/40 text-rose-700 dark:text-rose-400 font-medium border border-rose-200 dark:border-rose-800/40 transition-colors focus:outline-none focus:ring-2 focus:ring-rose-400"
          aria-label="Stop Camera (Shortcut: S)"
        >
          <Square className="w-4 h-4 fill-current" />
          <span>Stop (S)</span>
        </button>
      )}
    </div>
  );
};
