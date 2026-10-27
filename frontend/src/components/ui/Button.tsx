import type { ButtonHTMLAttributes, ReactNode } from 'react';
import { buttonClass, type Size, type Variant } from './buttonClass';

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  /** For toggle buttons: pressed state (also sets aria-pressed) */
  active?: boolean;
  children: ReactNode;
}

export function Button({ variant = 'secondary', size = 'md', active, className = '', children, ...rest }: Props) {
  return (
    <button
      type="button"
      aria-pressed={variant === 'toggle' ? !!active : undefined}
      className={`${buttonClass(variant, size, active)} ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}
