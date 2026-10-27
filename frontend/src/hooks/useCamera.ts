/**
 * SignTalk AI - Camera Management Hook
 * Phase 5: Robust camera lifecycle, permissions, and stream binding.
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import type { CameraStatus } from '../types/realtime';

export function useCamera() {
  const [status, setStatus] = useState<CameraStatus>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const startCamera = useCallback(async (deviceId?: string) => {
    setStatus('requesting_permission');
    setErrorMessage(null);

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus('error');
      setErrorMessage('Your browser does not support webcam access. Please use Chrome, Edge, or Firefox.');
      return;
    }

    // Switching cameras: release the previous one first
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    try {
      const constraints: MediaStreamConstraints = {
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 25 },
          deviceId: deviceId ? { exact: deviceId } : undefined,
          facingMode: deviceId ? undefined : 'user',
        },
        audio: false,
      };

      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
      } catch (err: any) {
        // A remembered camera that is no longer connected: fall back to the default one
        if (!deviceId || (err.name !== 'OverconstrainedError' && err.name !== 'NotFoundError')) throw err;
        stream = await navigator.mediaDevices.getUserMedia({ ...constraints, video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 25 }, facingMode: 'user' } });
      }
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play().catch((e) => console.error('Error playing camera stream:', e));
        };
      }

      setStatus('running');
    } catch (err: any) {
      console.error('Camera access error:', err);
      setStatus('error');
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setErrorMessage('Camera access was denied. Please allow camera permissions in your browser address bar.');
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setErrorMessage('No camera found. Please connect a webcam and try again.');
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        setErrorMessage('Camera is currently in use by another application (Teams, Zoom, etc.).');
      } else {
        setErrorMessage(`Camera initialization error: ${err.message || 'Unknown error'}`);
      }
    }
  }, []);

  const pauseCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getVideoTracks().forEach((track) => {
        track.enabled = false;
      });
      setStatus('paused');
    }
  }, []);

  const resumeCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getVideoTracks().forEach((track) => {
        track.enabled = true;
      });
      setStatus('running');
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setStatus('stopped');
  }, []);

  useEffect(() => {
    return () => {
      // Cleanup on unmount
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  return {
    videoRef,
    status,
    errorMessage,
    startCamera,
    pauseCamera,
    resumeCamera,
    stopCamera,
  };
}
