import { useEffect, useState } from 'react';

export type Route = 'home' | 'translator' | 'about' | 'contact';

export const ROUTES: { id: Route; label: string }[] = [
  { id: 'home', label: 'Home' },
  { id: 'translator', label: 'Live Translator' },
  { id: 'about', label: 'About' },
  { id: 'contact', label: 'Contact' },
];

/** Hash routes (#/translator) work on any static host without rewrite rules. */
export const hrefFor = (r: Route) => (r === 'home' ? '#/' : `#/${r}`);

function parse(hash: string): Route {
  const id = hash.replace(/^#\/?/, '').split(/[/?#]/)[0];
  return ROUTES.some((r) => r.id === id) ? (id as Route) : 'home';
}

export function useRoute() {
  const [route, setRoute] = useState<Route>(() => parse(window.location.hash));

  useEffect(() => {
    const onChange = () => {
      setRoute(parse(window.location.hash));
      window.scrollTo({ top: 0 });
    };
    window.addEventListener('hashchange', onChange);
    return () => window.removeEventListener('hashchange', onChange);
  }, []);

  return route;
}
