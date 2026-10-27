import { EyeOff, Lock, Server, Shield } from 'lucide-react';
import { LOCAL_INFERENCE } from '../../services/api';
import { Modal } from '../ui/Modal';
import { Button } from '../ui/Button';

export function PrivacyModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const items = [
    {
      icon: EyeOff,
      title: 'No video recording',
      text: 'Camera frames are only used in memory to extract hand and body landmarks, then discarded. Nothing is saved to disk.',
    },
    LOCAL_INFERENCE
      ? {
          icon: Server,
          title: 'Runs on your device',
          text: 'The landmark models and the sign classifier run inside this browser tab. Camera frames are never sent to a server.',
        }
      : {
          icon: Server,
          title: 'Local backend only',
          text: 'Frames are sent over your connection to the SignTalk backend for recognition. The backend does not store video.',
        },
    {
      icon: Lock,
      title: 'Temporary transcripts',
      text: 'Recognised signs and sentences exist only in this browser window and are cleared when you press Reset or close the tab.',
    },
  ];

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Privacy & video usage"
      icon={<Shield className="w-5 h-5" />}
      footer={
        <Button variant="primary" size="sm" onClick={onClose}>
          Understood
        </Button>
      }
    >
      <ul className="flex flex-col gap-3">
        {items.map(({ icon: Icon, title, text }) => (
          <li key={title} className="flex items-start gap-3 p-3 rounded-xl bg-surface-2 border border-line">
            <Icon className="w-5 h-5 text-accent-text shrink-0 mt-0.5" aria-hidden="true" />
            <div>
              <p className="font-semibold text-ink mb-0.5">{title}</p>
              <p className="text-xs leading-relaxed">{text}</p>
            </div>
          </li>
        ))}
      </ul>
    </Modal>
  );
}
