/**
 * Lists the video inputs and remembers the chosen one. Laptops often list a virtual camera
 * (e.g. Lenovo Smart Connect) before the real webcam, so the choice survives a reload.
 */

import { useCallback, useEffect, useState } from 'react';

const KEY = 'signtalk-camera';

export interface CameraDevice {
  deviceId: string;
  label: string;
}

function load(): string | undefined {
  try {
    return localStorage.getItem(KEY) || undefined;
  } catch {
    return undefined;
  }
}

export function useCameraDevices(cameraActive: boolean) {
  const [devices, setDevices] = useState<CameraDevice[]>([]);
  const [selectedId, setSelectedId] = useState<string | undefined>(load);

  // Labels only become readable once camera permission is granted, so list again when it starts
  useEffect(() => {
    const md = navigator.mediaDevices;
    if (!md?.enumerateDevices) return;
    let alive = true;
    const refresh = () =>
      md.enumerateDevices().then((all) => {
        if (!alive) return;
        setDevices(
          all
            .filter((d) => d.kind === 'videoinput' && d.deviceId)
            .map((d, i) => ({ deviceId: d.deviceId, label: d.label || `Camera ${i + 1}` })),
        );
      });
    refresh();
    md.addEventListener?.('devicechange', refresh);
    return () => {
      alive = false;
      md.removeEventListener?.('devicechange', refresh);
    };
  }, [cameraActive]);

  const select = useCallback((id: string) => {
    setSelectedId(id);
    try {
      localStorage.setItem(KEY, id);
    } catch {
      // Not remembered across reloads, but the switch still happens
    }
  }, []);

  return { devices, selectedId, select };
}
