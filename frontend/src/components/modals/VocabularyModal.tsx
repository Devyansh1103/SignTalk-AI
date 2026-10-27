import { useMemo, useState } from 'react';
import { BookOpen, Search } from 'lucide-react';
import type { VocabularyItem } from '../../types/realtime';
import { Modal } from '../ui/Modal';
import { Button } from '../ui/Button';

interface Props {
  open: boolean;
  onClose: () => void;
  vocabulary: VocabularyItem[];
}

export function VocabularyModal({ open, onClose, vocabulary }: Props) {
  const [query, setQuery] = useState('');
  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    const sorted = [...vocabulary].sort((a, b) => a.english_translation.localeCompare(b.english_translation));
    return q
      ? sorted.filter((v) => v.english_translation.toLowerCase().includes(q) || v.label.includes(q) || v.hindi_translation.includes(q))
      : sorted;
  }, [vocabulary, query]);

  return (
    <Modal
      open={open}
      onClose={onClose}
      wide
      title={`Supported signs (${vocabulary.length})`}
      icon={<BookOpen className="w-5 h-5" />}
      footer={
        <Button variant="primary" size="sm" onClick={onClose}>
          Got it
        </Button>
      }
    >
      <p className="text-xs mb-3">
        Sign one word at a time and lower your hands after each sign. Pause briefly to finish a sentence; consecutive signs are
        combined into English and Hindi text.
      </p>
      <label className="relative block mb-3">
        <span className="sr-only">Search signs</span>
        <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-ink-3" aria-hidden="true" />
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search a word…"
          className="w-full pl-9 pr-3 py-2 rounded-xl bg-surface-2 border border-line text-sm text-ink placeholder:text-ink-3"
        />
      </label>
      <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {shown.map((v) => (
          <li key={v.class_id} className="px-3 py-2 rounded-xl bg-surface-2 border border-line flex items-center justify-between gap-2">
            <span className="font-medium text-ink">{v.english_translation}</span>
            <span className="text-ink-3">{v.hindi_translation}</span>
          </li>
        ))}
      </ul>
      {shown.length === 0 && <p className="text-center text-ink-3 py-6">No sign matches “{query}”.</p>}
    </Modal>
  );
}
