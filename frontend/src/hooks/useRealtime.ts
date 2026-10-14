/**
 * SignTalk AI - Real-Time Streaming Hook
 * Phase 5: Coordinates frame capture, WebSocket dispatch, and live prediction telemetry.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { RealtimeClient } from '../services/realtime';
import type {
  ServerRealtimeMessage,
  ConnectionStatus,
  FinalizedPhrase,
} from '../types/realtime';

export function useRealtime(videoRef: React.RefObject<HTMLVideoElement | null>, isCameraRunning: boolean) {
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [latestMessage, setLatestMessage] = useState<ServerRealtimeMessage | null>(null);
  const [transcript, setTranscript] = useState<FinalizedPhrase[]>([]);
  const [fps, setFps] = useState<number>(0);

  const clientRef = useRef<RealtimeClient | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const captureIntervalRef = useRef<number | null>(null);
  const isStreamingRef = useRef<boolean>(false);

  // Backpressure & Deduplication guards
  const inFlightRef = useRef<number>(0);
  const lastSendTimeRef = useRef<number>(0);
  const lastFinalizedTimestampRef = useRef<number | null>(null);

  // Initialize offscreen canvas for frame capture (320x240 optimized for CPU MediaPipe extraction)
  useEffect(() => {
    canvasRef.current = document.createElement('canvas');
    canvasRef.current.width = 320;
    canvasRef.current.height = 240;
  }, []);

  // Initialize Realtime WebSocket Client
  useEffect(() => {
    const client = new RealtimeClient();
    clientRef.current = client;

    const unsubStatus = client.subscribeStatus((st) => {
      setConnectionStatus(st);
    });

    const unsubMsg = client.subscribeMessage((msg) => {
      // Decrement in-flight tracker on server response
      inFlightRef.current = Math.max(0, inFlightRef.current - 1);

      setLatestMessage(msg);
      if (msg.fps) {
        setFps(msg.fps);
      }

      // Check for committed phrases (guaranteed idempotent by timestamp)
      if (
        msg.translation &&
        msg.translation.is_final &&
        msg.translation.text &&
        msg.timestamp !== lastFinalizedTimestampRef.current
      ) {
        lastFinalizedTimestampRef.current = msg.timestamp;
        setTranscript((prev) => [
          ...prev,
          {
            id: `phrase_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
            timestamp: msg.timestamp,
            english: msg.translation!.text,
            hindi: msg.translation!.hindi_text,
            tokens: msg.translation!.tokens,
            confidence: msg.translation!.confidence,
          },
        ]);
      }
    });

    client.connect();

    return () => {
      unsubStatus();
      unsubMsg();
      client.disconnect();
    };
  }, []);

  // Frame Capture Pacer Loop with Backpressure Gating
  const startStreaming = useCallback(() => {
    if (isStreamingRef.current) return;
    isStreamingRef.current = true;
    inFlightRef.current = 0;

    // Send start command to backend
    clientRef.current?.sendControl('start');

    captureIntervalRef.current = window.setInterval(() => {
      if (!isCameraRunning || !videoRef.current || !canvasRef.current || !clientRef.current) {
        return;
      }

      // Backpressure guard: allow up to 2 frames in-flight to prevent pipeline starvation
      // while preventing unbounded buffer buildup if processing spikes.
      const now = Date.now();
      if (inFlightRef.current >= 2) {
        if (now - lastSendTimeRef.current > 600) {
          inFlightRef.current = 0;
        } else {
          return;
        }
      }

      const video = videoRef.current;
      if (video.readyState < 2 || video.paused || video.ended) {
        return;
      }

      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Draw video frame to canvas at 320x240
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      inFlightRef.current += 1;
      lastSendTimeRef.current = now;

      // Convert to blob and send binary frame (quality 0.65 for fast network transfer)
      canvas.toBlob(
        (blob) => {
          if (blob && clientRef.current) {
            blob.arrayBuffer().then((buffer) => {
              clientRef.current?.sendFrameBytes(buffer);
            });
          } else {
            inFlightRef.current = Math.max(0, inFlightRef.current - 1);
          }
        },
        'image/jpeg',
        0.65
      );
    }, 40); // 25 FPS target cadence matching ST-GCN training data
  }, [isCameraRunning, videoRef]);

  const stopStreaming = useCallback(() => {
    isStreamingRef.current = false;
    inFlightRef.current = 0;
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    clientRef.current?.sendControl('stop');
  }, []);

  const pauseStreaming = useCallback(() => {
    isStreamingRef.current = false;
    inFlightRef.current = 0;
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    clientRef.current?.sendControl('pause');
  }, []);

  const resumeStreaming = useCallback(() => {
    startStreaming();
  }, [startStreaming]);

  const resetSession = useCallback(() => {
    lastFinalizedTimestampRef.current = null;
    inFlightRef.current = 0;
    clientRef.current?.sendControl('reset');
    setTranscript([]);
    setLatestMessage(null);
  }, []);

  // Sync streaming with camera running state
  useEffect(() => {
    if (isCameraRunning && connectionStatus === 'connected') {
      startStreaming();
    } else {
      stopStreaming();
    }
  }, [isCameraRunning, connectionStatus, startStreaming, stopStreaming]);

  return {
    connectionStatus,
    latestMessage,
    transcript,
    fps,
    startStreaming,
    stopStreaming,
    pauseStreaming,
    resumeStreaming,
    resetSession,
  };
}
