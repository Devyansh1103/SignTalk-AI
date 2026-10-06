import React from 'react';
import type { ConnectionStatus } from '../../types/realtime';
import { ConnectionBadge } from '../status/ConnectionBadge';
import { Hand, Sparkles } from 'lucide-react';

interface Props {
  connectionStatus: ConnectionStatus;
}

export const Header: React.FC<Props> = ({ connectionStatus }) => {
  return (
    <header className="w-full border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md sticky top-0 z-30 transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand identity */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-teal-600 text-white flex items-center justify-center shadow-md shadow-teal-900/20">
            <Hand className="w-5 h-5" />
          </div>
          <div className="flex flex-col">
            <div className="flex items-center gap-2">
              <h1 className="text-base sm:text-lg font-black tracking-tight text-slate-900 dark:text-white">
                SignTalk AI
              </h1>
              <span className="hidden sm:inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-teal-50 dark:bg-teal-950/40 text-teal-700 dark:text-teal-300 border border-teal-200 dark:border-teal-800">
                <Sparkles className="w-3 h-3" />
                ISL Translation
              </span>
            </div>
            <span className="text-[11px] text-slate-500 dark:text-slate-400 hidden sm:inline">
              Real-Time Indian Sign Language Translation Platform
            </span>
          </div>
        </div>

        {/* Live connection badge */}
        <ConnectionBadge status={connectionStatus} />
      </div>
    </header>
  );
};
