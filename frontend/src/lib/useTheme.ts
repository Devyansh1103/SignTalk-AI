import { useCallback, useState } from 'react';

export type Theme = 'dark' | 'light';
const KEY = 'signtalk-theme';

/** Dark/light theme on <html data-theme>, remembered in localStorage (index.html applies it before paint). */
export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(() =>
    document.documentElement.dataset.theme === 'light' ? 'light' : 'dark',
  );

  const setTheme = useCallback((t: Theme) => {
    document.documentElement.dataset.theme = t;
    try {
      localStorage.setItem(KEY, t);
    } catch {
      // Storage blocked (private window): the theme still applies for this visit
    }
    setThemeState(t);
  }, []);

  return { theme, setTheme };
}
