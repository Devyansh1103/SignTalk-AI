import { useEffect, useId, useRef, type ReactNode } from 'react';
import { X } from 'lucide-react';

interface Props {
  open: boolean;
  onClose: () => void;
  title: string;
  icon?: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
}

export function Modal({ open, onClose, title, icon, children, footer, wide }: Props) {
  const titleId = useId();
  const panelRef = useRef<HTMLDivElement>(null);
  // The translator re-renders many times a second; keep the effect tied to `open` only
  const onCloseRef = useRef(onClose);
  useEffect(() => {
    onCloseRef.current = onClose;
  });

  useEffect(() => {
    if (!open) return;
    const previous = document.activeElement as HTMLElement | null;
    panelRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCloseRef.current();
    };
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('keydown', onKey);
      previous?.focus();
    };
  }, [open]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        onClick={(e) => e.stopPropagation()}
        className={`card w-full ${wide ? 'max-w-2xl' : 'max-w-md'} max-h-[85vh] flex flex-col overflow-hidden outline-none`}
      >
        <div className="flex items-center justify-between gap-3 px-5 py-4 border-b border-line">
          <div className="flex items-center gap-2 text-accent-text">
            {icon}
            <h2 id={titleId} className="text-base font-semibold text-ink">
              {title}
            </h2>
          </div>
          <button type="button" onClick={onClose} className="p-1 rounded-lg text-ink-3 hover:text-ink hover:bg-surface-2" aria-label="Close">
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="p-5 overflow-y-auto text-sm text-ink-2">{children}</div>
        {footer && <div className="px-5 py-3 border-t border-line flex justify-end">{footer}</div>}
      </div>
    </div>
  );
}
