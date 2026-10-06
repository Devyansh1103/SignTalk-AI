import React from 'react';
import type { CameraStatus } from '../../types/realtime';
import { Camera, CameraOff, AlertCircle, RefreshCw } from 'lucide-react';
import { CameraGuide } from './CameraGuide';

interface Props {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  status: CameraStatus;
  errorMessage: string | null;
  fps: number;
  mirror: boolean;
  onStartCamera: () => void;
}

export const CameraView: React.FC<Props> = ({
  videoRef,
  status,
  errorMessage,
  fps,
  mirror,
  onStartCamera,
}) => {
  return (
    <div
      id="camera-view-container"
      className="relative w-full aspect-[4/3] bg-slate-900 rounded-2xl overflow-hidden shadow-lg border border-slate-700/50 flex items-center justify-center"
      aria-label="Live camera preview area"
    >
      {/* Video Element */}
      <video
        ref={videoRef}
        id="camera-video-feed"
        autoPlay
        playsInline
        muted
        className={`w-full h-full object-cover transition-transform duration-200 ${
          mirror ? '-scale-x-100' : ''
        } ${status === 'running' ? 'block' : 'hidden'}`}
      />

      {/* Camera Guide Overlay */}
      {status === 'running' && <CameraGuide />}

      {/* Floating Status & FPS Badge */}
      {status === 'running' && (
        <div className="absolute top-3 left-3 flex items-center gap-2 bg-slate-900/80 backdrop-blur-md px-2.5 py-1 rounded-lg border border-slate-700/60 text-xs font-mono text-slate-200">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span>LIVE</span>
          {fps > 0 && <span className="text-slate-400">| {fps.toFixed(1)} FPS</span>}
        </div>
      )}

      {/* Idle / Permission Prompt State */}
      {status === 'idle' && (
        <div className="flex flex-col items-center justify-center p-6 text-center text-slate-300">
          <div className="w-16 h-16 rounded-full bg-teal-500/10 border border-teal-500/30 flex items-center justify-center mb-4 text-teal-400">
            <Camera className="w-8 h-8" />
          </div>
          <h3 className="text-lg font-semibold text-white mb-2">Camera Access Required</h3>
          <p className="text-sm text-slate-400 max-w-sm mb-6">
            SignTalk AI needs your camera to recognize hand gestures and facial expressions.
            Video is analyzed for real-time sign recognition.
          </p>
          <button
            id="start-camera-button"
            onClick={onStartCamera}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-teal-600 hover:bg-teal-500 text-white font-medium shadow-md shadow-teal-900/20 transition-all focus:outline-none focus:ring-2 focus:ring-teal-400"
          >
            <Camera className="w-4 h-4" />
            Enable Camera
          </button>
        </div>
      )}

      {/* Requesting Permission State */}
      {status === 'requesting_permission' && (
        <div className="flex flex-col items-center justify-center p-6 text-center text-slate-300">
          <RefreshCw className="w-10 h-10 text-teal-400 animate-spin mb-4" />
          <h3 className="text-lg font-medium text-white mb-1">Requesting Camera Access...</h3>
          <p className="text-sm text-slate-400">Please click &quot;Allow&quot; in the browser permission prompt.</p>
        </div>
      )}

      {/* Paused State Overlay */}
      {status === 'paused' && (
        <div className="absolute inset-0 bg-slate-900/80 backdrop-blur-sm flex flex-col items-center justify-center text-center p-6">
          <CameraOff className="w-10 h-10 text-slate-400 mb-3" />
          <h4 className="text-base font-medium text-white mb-1">Camera Stream Paused</h4>
          <p className="text-xs text-slate-400">Inference is currently paused.</p>
        </div>
      )}

      {/* Error State */}
      {status === 'error' && (
        <div className="flex flex-col items-center justify-center p-6 text-center text-slate-300 max-w-md">
          <div className="w-14 h-14 rounded-full bg-rose-500/10 border border-rose-500/30 flex items-center justify-center mb-4 text-rose-400">
            <AlertCircle className="w-7 h-7" />
          </div>
          <h3 className="text-base font-semibold text-white mb-2">Camera Unavailable</h3>
          <p className="text-sm text-rose-300/90 mb-5">{errorMessage || 'Unable to start camera stream.'}</p>
          <button
            id="retry-camera-button"
            onClick={onStartCamera}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-medium border border-slate-700 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            Try Again
          </button>
        </div>
      )}
    </div>
  );
};
