/**
 * SignTalk AI - In-browser recognition hook (no backend).
 * Same interface as useRealtime, but MediaPipe, the sign pipeline and the ST-GCN all run on the
 * viewer's device (src/engine), so the app can be hosted as a static site.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import type { ServerRealtimeMessage, ConnectionStatus, FinalizedPhrase } from '../types/realtime';
import { SignPipeline, type PipelineSettings, type VocabEntry } from '../engine/pipeline';
import { LandmarkExtractor } from '../engine/landmarks';
import { SignClassifier } from '../engine/classifier';

const MODEL_BASE = `${import.meta.env.BASE_URL}models`;
// The backend received 640x480 frames, and aspect correction + the parity tests assume them
const FRAME_W = 640;
const FRAME_H = 480;

interface Engine {
  pipeline: SignPipeline;
  extractor: LandmarkExtractor;
  classifier: SignClassifier;
}

async function loadEngine(): Promise<Engine> {
  const [settings, vocab] = await Promise.all([
    fetch(`${MODEL_BASE}/pipeline.json`).then((r) => r.json() as Promise<PipelineSettings & { mediapipe: Record<string, number> }>),
    fetch(`${MODEL_BASE}/vocab.json`).then((r) => r.json() as Promise<VocabEntry[]>),
  ]);
  const [extractor, classifier] = await Promise.all([
    LandmarkExtractor.create({
      poseConfidence: settings.mediapipe.pose_confidence,
      handConfidence: settings.mediapipe.hand_confidence,
      handTrackingConfidence: settings.mediapipe.hand_tracking_confidence,
      delegate: 'GPU',
    }),
    SignClassifier.create(settings.sequence_length),
  ]);
  return { pipeline: new SignPipeline(settings, vocab), extractor, classifier };
}

export function useLocalRealtime(videoRef: React.RefObject<HTMLVideoElement | null>, isCameraRunning: boolean) {
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('connecting');
  const [latestMessage, setLatestMessage] = useState<ServerRealtimeMessage | null>(null);
  const [transcript, setTranscript] = useState<FinalizedPhrase[]>([]);
  const [fps, setFps] = useState<number>(0);

  const engineRef = useRef<Engine | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const runningRef = useRef(false);
  const lastFinalizedTimestampRef = useRef<number | null>(null);

  useEffect(() => {
    const canvas = document.createElement('canvas');
    canvas.width = FRAME_W;
    canvas.height = FRAME_H;
    canvasRef.current = canvas;

    let cancelled = false;
    loadEngine()
      .then((engine) => {
        if (cancelled) return engine.extractor.close();
        engineRef.current = engine;
        setConnectionStatus('connected');
      })
      .catch((e) => {
        console.error('Failed to load the on-device AI models:', e);
        if (!cancelled) setConnectionStatus('error');
      });
    return () => {
      cancelled = true;
      runningRef.current = false;
      engineRef.current?.extractor.close();
      engineRef.current = null;
    };
  }, []);

  /** One frame: landmarks -> pipeline -> (when a sign ends) classifier, then a UI message. */
  const processFrame = useCallback(async () => {
    const engine = engineRef.current, video = videoRef.current, canvas = canvasRef.current;
    if (!engine || !video || !canvas || video.readyState < 2 || video.paused || video.ended) return;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    const t0 = performance.now();
    ctx.drawImage(video, 0, 0, FRAME_W, FRAME_H);
    const raw = engine.extractor.extract(canvas);
    const t1 = performance.now();
    const now = t1 / 1000;
    const { pipeline } = engine;
    const res = pipeline.step(pipeline.makeFrame(raw, now, FRAME_W, FRAME_H));
    const latencies: Record<string, number> = { mediapipe: t1 - t0 };
    if (res.segment) {
      const t2 = performance.now();
      const probs = await engine.classifier.classify(res.segment.input);
      latencies.stgcn_inference = performance.now() - t2;
      const pred = pipeline.acceptPrediction(probs, res.segment.frames, now);
      // Same line as the backend's [SEGMENT] log, for checking what the model saw
      const top3 = pred.top_k.map(([, l, c]) => `${l}:${c.toFixed(2)}`).join(', ');
      console.info(`[SEGMENT] ${pred.accepted ? pred.label : 'rejected'} top3=[${top3}]`);
    }
    latencies.end_to_end = performance.now() - t0;

    const translation = pipeline.translationFor(now);
    const p = pipeline.latestPrediction;
    const msg: ServerRealtimeMessage = {
      type: 'prediction',
      session_id: 'local',
      timestamp: now,
      pipeline_state: 'RUNNING',
      prediction: p
        ? { class_id: p.class_id, label: p.label, gloss: p.gloss, confidence: p.confidence, input_quality: p.input_quality, top_k: p.top_k }
        : undefined,
      smoothed_label: pipeline.latestLabel,
      state_machine: pipeline.displayState(now),
      sign_sequence: pipeline.sequence,
      translation: translation ?? undefined,
      stage_latencies_ms: latencies,
      fps: pipeline.fps,
      is_valid_input: res.frame.isValid,
    };
    setLatestMessage(msg);
    setFps(pipeline.fps);

    if (translation?.is_final && translation.text && msg.timestamp !== lastFinalizedTimestampRef.current) {
      lastFinalizedTimestampRef.current = msg.timestamp;
      setTranscript((prev) => [
        ...prev,
        {
          id: `phrase_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
          timestamp: msg.timestamp,
          english: translation.text,
          hindi: translation.hindi_text,
          tokens: translation.tokens,
          confidence: translation.confidence,
        },
      ]);
    }
  }, [videoRef]);

  // Process frames back to back: the next frame starts as soon as the previous one is done
  const startStreaming = useCallback(() => {
    if (runningRef.current) return;
    runningRef.current = true;
    const loop = async () => {
      while (runningRef.current) {
        try {
          await processFrame();
        } catch (e) {
          console.error('Frame processing failed:', e);
        }
        // Yield to the browser so the UI can repaint between frames
        await new Promise((r) => requestAnimationFrame(() => r(null)));
      }
    };
    loop();
  }, [processFrame]);

  const stopStreaming = useCallback(() => {
    runningRef.current = false;
  }, []);

  const pauseStreaming = useCallback(() => {
    runningRef.current = false;
  }, []);

  const resumeStreaming = useCallback(() => {
    startStreaming();
  }, [startStreaming]);

  const resetSession = useCallback(() => {
    lastFinalizedTimestampRef.current = null;
    engineRef.current?.pipeline.reset();
    setTranscript([]);
    setLatestMessage(null);
  }, []);

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
