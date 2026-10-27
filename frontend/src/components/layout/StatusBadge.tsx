import type { ConnectionStatus } from '../../types/realtime';

/** Model state from the recognition hook: on-device models loading/ready, or the backend connection. */
export function StatusBadge({ status, local }: { status: ConnectionStatus; local: boolean }) {
  const cfg = {
    connected: { label: local ? 'AI Ready' : 'AI Connected', tone: 'text-ok border-ok/30 bg-ok/10', dot: 'bg-ok' },
    processing: { label: 'Processing', tone: 'text-accent-text border-accent/30 bg-accent-soft', dot: 'bg-accent' },
    connecting: { label: local ? 'Loading AI…' : 'Connecting…', tone: 'text-warn border-warn/30 bg-warn/10', dot: 'bg-warn animate-pulse' },
    paused: { label: 'Paused', tone: 'text-ink-3 border-line bg-surface-2', dot: 'bg-ink-3' },
    error: { label: local ? 'AI failed to load' : 'Connection error', tone: 'text-bad border-bad/30 bg-bad/10', dot: 'bg-bad' },
    disconnected: { label: 'Offline', tone: 'text-ink-3 border-line bg-surface-2', dot: 'bg-ink-3' },
  }[status];

  return (
    <span
      role="status"
      aria-live="polite"
      className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border whitespace-nowrap ${cfg.tone}`}
    >
      <span className={`w-2 h-2 rounded-full ${cfg.dot}`} aria-hidden="true" />
      {cfg.label}
    </span>
  );
}
