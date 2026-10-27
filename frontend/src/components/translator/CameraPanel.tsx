import { useEffect, type RefObject } from 'react';
import { AlertCircle, Camera, CameraOff, ChevronDown, FlipHorizontal, LoaderCircle, Pause, Play, RotateCcw, Square } from 'lucide-react';
import type { CameraStatus, ConnectionStatus } from '../../types/realtime';
import type { CameraDevice } from '../../hooks/useCameraDevices';
import { LOCAL_INFERENCE } from '../../services/api';
import { Button } from '../ui/Button';

interface Props {
  videoRef: RefObject<HTMLVideoElement | null>;
  cameraStatus: CameraStatus;
  cameraError: string | null;
  modelStatus: ConnectionStatus;
  signing: boolean;
  mirror: boolean;
  devices: CameraDevice[];
  selectedDeviceId?: string;
  onSelectDevice: (id: string) => void;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onStop: () => void;
  onReset: () => void;
  onToggleMirror: () => void;
}

function Overlay({ icon, title, text, children, tone = 'accent' }: {
  icon: React.ReactNode;
  title: string;
  text?: string;
  children?: React.ReactNode;
  tone?: 'accent' | 'bad';
}) {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-6 bg-[#060b1d]/85 backdrop-blur-sm">
      <div
        className={`w-14 h-14 rounded-full flex items-center justify-center mb-4 border ${
          tone === 'bad' ? 'bg-rose-500/10 border-rose-400/30 text-rose-300' : 'bg-indigo-500/15 border-indigo-400/30 text-indigo-200'
        }`}
      >
        {icon}
      </div>
      <h3 className="text-base font-semibold text-white mb-1.5">{title}</h3>
      {text && <p className="text-sm text-slate-300 max-w-sm mb-5">{text}</p>}
      {children}
    </div>
  );
}

export function CameraPanel({ videoRef, ...p }: Props) {
  const { cameraStatus: cs, modelStatus: ms } = p;
  const running = cs === 'running';
  const paused = cs === 'paused';
  const off = cs === 'idle' || cs === 'stopped' || cs === 'error';

  // Keyboard shortcuts (same as before): Space = start / pause / resume, R = reset, S = stop
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (['INPUT', 'TEXTAREA', 'SELECT'].includes(t.tagName) || t.isContentEditable)) return;
      if (document.querySelector('[role="dialog"]')) return;
      if (e.code === 'Space') {
        e.preventDefault();
        if (running) p.onPause();
        else if (paused) p.onResume();
        else if (off) p.onStart();
      } else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        p.onReset();
      } else if (e.key === 's' || e.key === 'S') {
        e.preventDefault();
        if (!off) p.onStop();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [running, paused, off, p]);

  return (
    <section aria-label="Camera" className="card overflow-hidden">
      <div className="relative aspect-[4/3] bg-[#050914]">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          aria-label="Live camera preview"
          className={`w-full h-full object-cover ${p.mirror ? '-scale-x-100' : ''} ${running || paused ? 'block' : 'invisible'}`}
        />

        {running && ms === 'connected' && (
          <div className="absolute top-3 left-3 flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-emerald-500/90 text-white text-xs font-semibold">
              <span className="w-1.5 h-1.5 rounded-full bg-white" aria-hidden="true" />
              Live
            </span>
            {p.signing && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-indigo-500/90 text-white text-xs font-semibold" role="status">
                Signing… lower your hands when done
              </span>
            )}
          </div>
        )}

        {/* Upper-body guide while waiting for a sign */}
        {running && ms === 'connected' && !p.signing && (
          <div className="absolute inset-[8%] border-2 border-dashed border-white/20 rounded-2xl pointer-events-none flex items-start justify-center" aria-hidden="true">
            <span className="mt-2 px-2 py-0.5 rounded bg-black/50 text-[11px] text-white/80">Keep your upper body and hands in the frame</span>
          </div>
        )}

        {(cs === 'idle' || cs === 'stopped') && (
          <Overlay
            icon={<Camera className="w-7 h-7" />}
            title={cs === 'idle' ? 'Camera access required' : 'Session stopped'}
            text={
              cs === 'idle'
                ? `SignTalk AI uses your camera to follow your hands. ${
                    LOCAL_INFERENCE ? 'Video stays on this device and is never recorded.' : 'Video is never recorded.'
                  }`
                : 'The camera is off. Start again whenever you are ready.'
            }
          >
            <Button variant="primary" onClick={p.onStart}>
              <Camera className="w-4 h-4" />
              {cs === 'idle' ? 'Enable camera' : 'Start camera'}
            </Button>
          </Overlay>
        )}

        {cs === 'requesting_permission' && (
          <Overlay
            icon={<LoaderCircle className="w-7 h-7 animate-spin" />}
            title="Starting camera…"
            text='If your browser asks, click "Allow" to share the camera.'
          />
        )}

        {cs === 'error' && (
          <Overlay icon={<AlertCircle className="w-7 h-7" />} tone="bad" title="Camera unavailable" text={p.cameraError || 'Unable to start the camera.'}>
            <Button variant="secondary" onClick={p.onStart}>
              <RotateCcw className="w-4 h-4" />
              Try again
            </Button>
          </Overlay>
        )}

        {running && ms === 'connecting' && (
          <Overlay
            icon={<LoaderCircle className="w-7 h-7 animate-spin" />}
            title="Loading AI models…"
            text="The first visit downloads the models; later visits load them from the browser cache."
          />
        )}

        {(running || paused) && ms === 'error' && (
          <Overlay
            icon={<AlertCircle className="w-7 h-7" />}
            tone="bad"
            title="AI models failed to load"
            text="Check your internet connection and reload the page. Details are in the browser console (F12)."
          />
        )}

        {paused && ms !== 'error' && (
          <Overlay icon={<CameraOff className="w-7 h-7" />} title="Paused" text="Recognition is paused. Press Resume or Space to continue." />
        )}
      </div>

      {/* Controls */}
      <div role="toolbar" aria-label="Session controls" className="flex flex-wrap items-center gap-2 p-3 border-t border-line">
        <label className="relative flex-1 min-w-[12rem]">
          <span className="sr-only">Camera</span>
          <Camera className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-ink-3 pointer-events-none" aria-hidden="true" />
          <select
            value={p.selectedDeviceId && p.devices.some((d) => d.deviceId === p.selectedDeviceId) ? p.selectedDeviceId : ''}
            onChange={(e) => p.onSelectDevice(e.target.value)}
            disabled={p.devices.length === 0}
            className="w-full appearance-none pl-9 pr-8 py-2.5 rounded-xl bg-surface-2 border border-line text-sm text-ink truncate disabled:opacity-60"
          >
            <option value="">{p.devices.length ? 'Default camera' : 'No camera listed yet'}</option>
            {p.devices.map((d) => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label}
              </option>
            ))}
          </select>
          <ChevronDown className="w-4 h-4 absolute right-3 top-1/2 -translate-y-1/2 text-ink-3 pointer-events-none" aria-hidden="true" />
        </label>

        {off && (
          <Button variant="primary" onClick={p.onStart} aria-keyshortcuts="Space">
            <Play className="w-4 h-4" />
            Start
          </Button>
        )}
        {running && (
          <Button variant="secondary" onClick={p.onPause} aria-keyshortcuts="Space">
            <Pause className="w-4 h-4" />
            Pause
          </Button>
        )}
        {paused && (
          <Button variant="primary" onClick={p.onResume} aria-keyshortcuts="Space">
            <Play className="w-4 h-4" />
            Resume
          </Button>
        )}
        {!off && (
          <Button variant="danger" onClick={p.onStop} aria-keyshortcuts="S">
            <Square className="w-4 h-4 fill-current" />
            Stop
          </Button>
        )}
        <Button variant="toggle" active={p.mirror} onClick={p.onToggleMirror} title="Mirror the preview">
          <FlipHorizontal className="w-4 h-4" />
          Flip
        </Button>
        <Button variant="toggle" onClick={p.onReset} aria-keyshortcuts="R" title="Clear signs and transcript">
          <RotateCcw className="w-4 h-4" />
          Reset
        </Button>
      </div>
      <p className="px-3 pb-3 text-[11px] text-ink-3">
        Shortcuts: <kbd className="font-mono">Space</kbd> start/pause · <kbd className="font-mono">R</kbd> reset · <kbd className="font-mono">S</kbd> stop
      </p>
    </section>
  );
}
