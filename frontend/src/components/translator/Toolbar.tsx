import { BookOpen, Eye, FlipHorizontal, Gauge, Shield, Type } from 'lucide-react';
import type { AccessibilitySettings } from '../../types/realtime';
import { Button } from '../ui/Button';

interface Props {
  settings: AccessibilitySettings;
  onChange: (updater: (s: AccessibilitySettings) => AccessibilitySettings) => void;
  onOpenVocabulary: () => void;
  onOpenPrivacy: () => void;
}

export function Toolbar({ settings, onChange, onOpenVocabulary, onOpenPrivacy }: Props) {
  const toggle = (key: 'largeText' | 'highContrast' | 'mirrorPreview' | 'showDebugMetrics', label: string, Icon: typeof Type) => (
    <Button variant="toggle" size="sm" active={settings[key]} onClick={() => onChange((s) => ({ ...s, [key]: !s[key] }))}>
      <Icon className="w-3.5 h-3.5" aria-hidden="true" />
      {label}
    </Button>
  );

  return (
    <div role="toolbar" aria-label="Display and accessibility settings" className="flex flex-wrap items-center justify-between gap-2">
      <div className="flex flex-wrap items-center gap-2">
        {toggle('largeText', 'Large Text', Type)}
        {toggle('highContrast', 'High Contrast', Eye)}
        {toggle('mirrorPreview', 'Mirror Video', FlipHorizontal)}
        {toggle('showDebugMetrics', 'Telemetry', Gauge)}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="toggle" size="sm" onClick={onOpenVocabulary}>
          <BookOpen className="w-3.5 h-3.5" aria-hidden="true" />
          Supported Signs
        </Button>
        <Button variant="toggle" size="sm" onClick={onOpenPrivacy}>
          <Shield className="w-3.5 h-3.5" aria-hidden="true" />
          Privacy
        </Button>
      </div>
    </div>
  );
}
