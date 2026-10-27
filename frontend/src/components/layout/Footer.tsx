import { Shield } from 'lucide-react';
import { PROJECT } from '../../lib/project';
import { GithubIcon } from '../ui/GithubIcon';

export function Footer({ onOpenPrivacy }: { onOpenPrivacy: () => void }) {
  return (
    <footer className="border-t border-line mt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 flex flex-col md:flex-row items-center justify-between gap-3 text-xs text-ink-3">
        <p className="flex flex-wrap justify-center md:justify-start items-center gap-x-3 gap-y-1 text-center">
          <span className="font-medium text-ink-2">{PROJECT.name}</span>
          <span aria-hidden="true">|</span>
          <span>{PROJECT.institution} Academic Project</span>
          <span aria-hidden="true">|</span>
          <span>Spatial-Temporal Graph Convolutional Network</span>
        </p>
        <div className="flex items-center gap-5">
          <a href={PROJECT.repoUrl} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1.5 hover:text-ink">
            <GithubIcon className="w-3.5 h-3.5" />
            Project on GitHub
          </a>
          <button type="button" onClick={onOpenPrivacy} className="inline-flex items-center gap-1.5 hover:text-ink">
            <Shield className="w-3.5 h-3.5" aria-hidden="true" />
            Privacy
          </button>
        </div>
      </div>
    </footer>
  );
}
