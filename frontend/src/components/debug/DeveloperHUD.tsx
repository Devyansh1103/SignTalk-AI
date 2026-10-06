import React from 'react';
import type { ServerRealtimeMessage } from '../../types/realtime';
import { Cpu, Zap, Activity } from 'lucide-react';

interface Props {
  message: ServerRealtimeMessage | null;
  fps: number;
}

export const DeveloperHUD: React.FC<Props> = ({ message, fps }) => {
  if (!message) return null;

  const latencies = message.stage_latencies_ms || {};
  const stgcnMs = latencies.stgcn_inference || 0;
  const e2eMs = latencies.end_to_end || 0;

  return (
    <div
      id="developer-telemetry-hud"
      className="p-4 rounded-2xl bg-slate-900 border border-slate-800 text-slate-300 font-mono text-xs flex flex-col gap-2.5 shadow-md"
    >
      <div className="flex items-center justify-between text-slate-400 pb-1 border-b border-slate-800">
        <div className="flex items-center gap-1.5 font-semibold uppercase text-[11px] text-teal-400">
          <Activity className="w-3.5 h-3.5" />
          <span>Inference Telemetry</span>
        </div>
        <span className="text-[10px]">Session: {message.session_id}</span>
      </div>

      <div className="grid grid-cols-3 gap-2 py-1">
        <div className="p-2 rounded-lg bg-slate-800/60 border border-slate-700/50 flex flex-col">
          <span className="text-[10px] text-slate-400 uppercase">Throughput</span>
          <span className="text-sm font-bold text-white flex items-center gap-1">
            <Zap className="w-3 h-3 text-amber-400" />
            {fps.toFixed(1)} FPS
          </span>
        </div>

        <div className="p-2 rounded-lg bg-slate-800/60 border border-slate-700/50 flex flex-col">
          <span className="text-[10px] text-slate-400 uppercase">ST-GCN Latency</span>
          <span className="text-sm font-bold text-teal-300 flex items-center gap-1">
            <Cpu className="w-3 h-3" />
            {stgcnMs.toFixed(1)} ms
          </span>
        </div>

        <div className="p-2 rounded-lg bg-slate-800/60 border border-slate-700/50 flex flex-col">
          <span className="text-[10px] text-slate-400 uppercase">End-to-End</span>
          <span className="text-sm font-bold text-emerald-300">
            {e2eMs.toFixed(1)} ms
          </span>
        </div>
      </div>

      {/* Stage Breakdown Chips */}
      <div className="flex flex-wrap gap-1.5 text-[10px] text-slate-400 pt-1">
        {Object.entries(latencies).map(([stage, ms]) => (
          <span key={stage} className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700">
            {stage}: {ms.toFixed(1)}ms
          </span>
        ))}
      </div>
    </div>
  );
};
