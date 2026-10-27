import { Gauge } from 'lucide-react';
import type { ServerRealtimeMessage } from '../../types/realtime';

/** Per-stage timings of the latest frame (the Telemetry toggle). */
export function TelemetryPanel({ message, fps }: { message: ServerRealtimeMessage | null; fps: number }) {
  const lat = message?.stage_latencies_ms ?? {};
  return (
    <section aria-labelledby="telemetry-title" className="card p-4 font-mono text-xs">
      <div className="flex items-center justify-between mb-2 text-ink-3">
        <h2 id="telemetry-title" className="flex items-center gap-1.5 font-sans text-sm font-medium text-ink-2">
          <Gauge className="w-4 h-4 text-accent-text" aria-hidden="true" />
          Inference telemetry
        </h2>
        <span>session: {message?.session_id ?? '—'}</span>
      </div>
      {message ? (
        <div className="flex flex-wrap gap-1.5">
          <span className="px-2 py-1 rounded-lg bg-surface-2 border border-line text-ink">fps: {fps.toFixed(1)}</span>
          {Object.entries(lat).map(([stage, ms]) => (
            <span key={stage} className="px-2 py-1 rounded-lg bg-surface-2 border border-line text-ink-2">
              {stage}: {ms.toFixed(1)} ms
            </span>
          ))}
          <span className="px-2 py-1 rounded-lg bg-surface-2 border border-line text-ink-2">
            valid input: {message.is_valid_input ? 'yes' : 'no'}
          </span>
        </div>
      ) : (
        <p className="text-ink-3">No frames processed yet.</p>
      )}
    </section>
  );
}
