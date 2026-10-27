import { useState } from 'react';
import { Menu, X } from 'lucide-react';
import type { ConnectionStatus } from '../../types/realtime';
import { ROUTES, hrefFor, type Route } from '../../lib/useRoute';
import type { Theme } from '../../lib/useTheme';
import { Logo } from './Logo';
import { ThemeToggle } from './ThemeToggle';
import { StatusBadge } from './StatusBadge';

interface Props {
  route: Route;
  theme: Theme;
  onThemeChange: (t: Theme) => void;
  status: ConnectionStatus;
  local: boolean;
}

export function Header({ route, theme, onThemeChange, status, local }: Props) {
  // The mobile menu belongs to the page it was opened on, so navigating closes it
  const [openOn, setOpenOn] = useState<Route | null>(null);
  const open = openOn === route;
  const setOpen = (fn: (o: boolean) => boolean) => setOpenOn(fn(open) ? route : null);

  const link = (id: Route, label: string, mobile = false) => (
    <a
      key={id}
      href={hrefFor(id)}
      aria-current={route === id ? 'page' : undefined}
      className={
        mobile
          ? `block px-3 py-2.5 rounded-lg text-sm font-medium ${route === id ? 'bg-accent-soft text-accent-text' : 'text-ink-2 hover:bg-surface-2'}`
          : `relative py-2 text-sm font-medium transition-colors ${
              route === id
                ? 'text-ink after:absolute after:left-0 after:right-0 after:-bottom-0.5 after:h-0.5 after:rounded-full after:bg-gradient-accent'
                : 'text-ink-2 hover:text-ink'
            }`
      }
    >
      {label}
    </a>
  );

  return (
    <header className="sticky top-0 z-40 border-b border-line bg-bg/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
        <Logo />

        <nav aria-label="Main" className="hidden md:flex items-center gap-8">
          {ROUTES.map((r) => link(r.id, r.label))}
        </nav>

        <div className="flex items-center gap-2">
          <div className="hidden sm:block">
            <ThemeToggle theme={theme} onChange={onThemeChange} />
          </div>
          <StatusBadge status={status} local={local} />
          <button
            type="button"
            className="md:hidden p-2 rounded-lg text-ink-2 hover:bg-surface-2"
            aria-label={open ? 'Close menu' : 'Open menu'}
            aria-expanded={open}
            aria-controls="mobile-nav"
            onClick={() => setOpen((o) => !o)}
          >
            {open ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {open && (
        <nav id="mobile-nav" aria-label="Main" className="md:hidden border-t border-line bg-bg px-4 py-3 flex flex-col gap-1">
          {ROUTES.map((r) => link(r.id, r.label, true))}
          <div className="sm:hidden pt-3 mt-2 border-t border-line">
            <ThemeToggle theme={theme} onChange={onThemeChange} />
          </div>
        </nav>
      )}
    </header>
  );
}
