import React from 'react';
import type { ConnectionStatus } from '../../types/realtime';
import { Wifi, WifiOff, RefreshCw, AlertCircle } from 'lucide-react';

interface Props {
  status: ConnectionStatus;
}

export const ConnectionBadge: React.FC<Props> = ({ status }) => {
  const getBadgeConfig = () => {
    switch (status) {
      case 'connected':
        return {
          label: 'AI Connected',
          className: 'bg-emerald-500/15 text-emerald-700 border-emerald-500/30 dark:text-emerald-400',
          dot: 'bg-emerald-500 animate-pulse',
          icon: <Wifi className="w-3.5 h-3.5" />,
        };
      case 'processing':
        return {
          label: 'Processing',
          className: 'bg-teal-500/15 text-teal-700 border-teal-500/30 dark:text-teal-400',
          dot: 'bg-teal-500 animate-ping',
          icon: <RefreshCw className="w-3.5 h-3.5 animate-spin" />,
        };
      case 'connecting':
        return {
          label: 'Connecting...',
          className: 'bg-amber-500/15 text-amber-700 border-amber-500/30 dark:text-amber-400',
          dot: 'bg-amber-500 animate-bounce',
          icon: <RefreshCw className="w-3.5 h-3.5 animate-spin" />,
        };
      case 'paused':
        return {
          label: 'Paused',
          className: 'bg-slate-500/15 text-slate-700 border-slate-500/30 dark:text-slate-400',
          dot: 'bg-slate-400',
          icon: <Wifi className="w-3.5 h-3.5" />,
        };
      case 'error':
        return {
          label: 'Connection Error',
          className: 'bg-rose-500/15 text-rose-700 border-rose-500/30 dark:text-rose-400',
          dot: 'bg-rose-500',
          icon: <AlertCircle className="w-3.5 h-3.5" />,
        };
      case 'disconnected':
      default:
        return {
          label: 'Offline',
          className: 'bg-slate-500/15 text-slate-600 border-slate-400/30 dark:text-slate-400',
          dot: 'bg-slate-400',
          icon: <WifiOff className="w-3.5 h-3.5" />,
        };
    }
  };

  const config = getBadgeConfig();

  return (
    <div
      id="connection-status-badge"
      className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border transition-colors ${config.className}`}
      role="status"
      aria-live="polite"
    >
      <span className={`w-2 h-2 rounded-full ${config.dot}`} aria-hidden="true" />
      <span>{config.label}</span>
    </div>
  );
};
