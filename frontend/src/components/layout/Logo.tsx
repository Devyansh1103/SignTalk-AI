import { Hand } from 'lucide-react';
import { PROJECT } from '../../lib/project';

export function Logo({ compact = false }: { compact?: boolean }) {
  return (
    <a href="#/" className="flex items-center gap-3 shrink-0" aria-label={`${PROJECT.name} home`}>
      <span className="w-10 h-10 rounded-xl bg-gradient-accent text-white flex items-center justify-center shadow-lg shadow-indigo-900/30">
        <Hand className="w-5 h-5" aria-hidden="true" />
      </span>
      <span className="flex flex-col leading-tight">
        <span className="text-lg font-bold tracking-tight text-ink">{PROJECT.name}</span>
        {!compact && <span className="hidden sm:block text-[11px] text-ink-3">{PROJECT.tagline}</span>}
      </span>
    </a>
  );
}
