import { Moon, Sun } from 'lucide-react';
import type { Theme } from '../../lib/useTheme';

export function ThemeToggle({ theme, onChange }: { theme: Theme; onChange: (t: Theme) => void }) {
  const option = (t: Theme, label: string, Icon: typeof Sun) => (
    <button
      type="button"
      onClick={() => onChange(t)}
      aria-pressed={theme === t}
      className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-full text-xs font-medium transition-colors ${
        theme === t ? 'bg-accent-soft text-accent-text' : 'text-ink-3 hover:text-ink'
      }`}
    >
      <Icon className="w-3.5 h-3.5" aria-hidden="true" />
      {label}
    </button>
  );

  return (
    <div role="group" aria-label="Colour theme" className="inline-flex items-center p-0.5 rounded-full border border-line bg-surface">
      {option('dark', 'Dark', Moon)}
      {option('light', 'Light', Sun)}
    </div>
  );
}
