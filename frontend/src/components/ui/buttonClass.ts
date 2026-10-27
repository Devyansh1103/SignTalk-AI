export type Variant = 'primary' | 'secondary' | 'danger' | 'ghost' | 'toggle';
export type Size = 'sm' | 'md' | 'lg';

const base =
  'inline-flex items-center justify-center gap-2 rounded-xl font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed';

const variants: Record<Variant, string> = {
  primary: 'bg-gradient-accent text-white shadow-lg shadow-indigo-900/25 hover:brightness-110',
  secondary: 'bg-surface border border-line-strong text-ink hover:bg-surface-2',
  danger: 'bg-bad text-white hover:brightness-110',
  ghost: 'text-ink-2 hover:text-ink hover:bg-surface-2',
  toggle: 'bg-surface border border-line text-ink-2 hover:text-ink hover:bg-surface-2',
};

const sizes: Record<Size, string> = {
  sm: 'px-3 py-1.5 text-xs',
  md: 'px-4 py-2.5 text-sm',
  lg: 'px-6 py-3 text-sm',
};

/** Class string for buttons and button-looking links. */
export function buttonClass(variant: Variant = 'secondary', size: Size = 'md', active = false) {
  const look = variant === 'toggle' && active ? 'bg-accent-soft border border-accent/50 text-accent-text' : variants[variant];
  return `${base} ${sizes[size]} ${look}`;
}
