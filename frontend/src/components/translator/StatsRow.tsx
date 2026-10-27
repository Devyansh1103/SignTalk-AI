import { Activity, Cpu, Layers, Timer } from 'lucide-react';
import type { ConnectionStatus } from '../../types/realtime';

interface Props {
  /** ST-GCN time for the last classified sign, ms (null until the first sign) */
  inferenceMs: number | null;
  fps: number;
  running: boolean;
  classes: number;
  modelStatus: ConnectionStatus;
  local: boolean;
}

/** Live numbers only: anything not measured yet shows a dash. */
export function StatsRow({ inferenceMs, fps, running, classes, modelStatus, local }: Props) {
  const model =
    modelStatus === 'connected' || modelStatus === 'processing'
      ? 'Ready'
      : modelStatus === 'connecting'
      ? 'Loading…'
      : modelStatus === 'error'
      ? 'Failed'
      : 'Offline';

  const stats = [
    { icon: Timer, value: inferenceMs !== null ? `${inferenceMs.toFixed(0)} ms` : '—', label: 'Inference', title: 'ST-GCN time for the last sign' },
    { icon: Activity, value: running && fps > 0 ? `${fps.toFixed(0)} FPS` : '—', label: 'Live FPS', title: 'Frames processed per second' },
    { icon: Layers, value: classes ? String(classes) : '—', label: 'Sign classes', title: 'Words the model knows' },
    { icon: Cpu, value: model, label: 'ST-GCN', title: local ? 'Model runs in this browser' : 'Model runs on the server' },
  ];

  return (
    <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
      {stats.map(({ icon: Icon, value, label, title }) => (
        <div key={label} title={title} className="card px-4 py-3 flex items-center gap-3">
          <span className="w-10 h-10 rounded-full bg-accent-soft text-accent-text flex items-center justify-center shrink-0">
            <Icon className="w-5 h-5" aria-hidden="true" />
          </span>
          <div className="min-w-0">
            <p className="text-base font-semibold text-ink truncate">{value}</p>
            <p className="text-xs text-ink-3 truncate">{label}</p>
          </div>
        </div>
      ))}
    </div>
  );
}
