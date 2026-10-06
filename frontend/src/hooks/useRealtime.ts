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

  // Initialize offscreen canvas for frame capture
  useEffect(() => {
    canvasRef.current = document.createElement('canvas');
    canvasRef.current.width = 640;
    canvasRef.current.height = 480;
  }, []);

  // Initialize Realtime WebSocket Client
  useEffect(() => {
    const client = new RealtimeClient();
    clientRef.current = client;

    const unsubStatus = client.subscribeStatus((st) => {
      setConnectionStatus(st);
    });

    const unsubMsg = client.subscribeMessage((msg) => {
      setLatestMessage(msg);
      if (msg.fps) {
        setFps(msg.fps);
      }

      // Check for committed phrases
      if (msg.translation && msg.translation.is_final && msg.translation.text) {
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

  // Frame Capture Pacer Loop (15–20 FPS to prevent socket saturation)
  const startStreaming = useCallback(() => {
    if (isStreamingRef.current) return;
    isStreamingRef.current = true;

    // Send start command to backend
    clientRef.current?.sendControl('start');

    captureIntervalRef.current = window.setInterval(() => {
      if (!isCameraRunning || !videoRef.current || !canvasRef.current || !clientRef.current) {
        return;
      }

      const video = videoRef.current;
      if (video.readyState < 2 || video.paused || video.ended) {
        return;
      }

      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Draw video frame to canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Convert to blob and send binary frame
      canvas.toBlob(
        (blob) => {
          if (blob && clientRef.current) {
            blob.arrayBuffer().then((buffer) => {
              clientRef.current?.sendFrameBytes(buffer);
            });
          }
        },
        'image/jpeg',
        0.80
      );
    }, 60); // ~16.6 FPS (60ms intervals)
  }, [isCameraRunning, videoRef]);

  const stopStreaming = useCallback(() => {
    isStreamingRef.current = false;
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    clientRef.current?.sendControl('stop');
  }, []);

  const pauseStreaming = useCallback(() => {
    isStreamingRef.current = false;
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
