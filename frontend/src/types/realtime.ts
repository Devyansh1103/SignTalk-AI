/**
 * SignTalk AI - Frontend Type Definitions
 * Phase 5: Strongly typed contracts for WebSocket streaming, API metadata, and UX state.
 */

export type ConnectionStatus =
  | 'connecting'
  | 'connected'
  | 'processing'
  | 'paused'
  | 'disconnected'
  | 'error';

export type CameraStatus =
  | 'idle'
  | 'requesting_permission'
  | 'running'
  | 'paused'
  | 'stopped'
  | 'error';

export type ConfidenceTier = 'high' | 'moderate' | 'low' | 'uncertain';

export interface TopKPrediction {
  class_id: number;
  label: string;
  confidence: number;
}

export interface PredictionPayload {
  class_id?: number;
  label?: string;
  gloss?: string;
  confidence: number;
  input_quality: number;
  top_k: [number, string, number][];
}

export interface TranslationPayload {
  text: string;
  hindi_text: string;
  confidence: number;
  tokens: string[];
  source_signs: string[];
  is_final: boolean;
}

export interface ServerRealtimeMessage {
  type: 'prediction' | 'status' | 'event' | 'error';
  session_id: string;
  timestamp: number;
  pipeline_state: 'RUNNING' | 'PAUSED' | 'STOPPED' | 'WARMING_UP' | 'ERROR';
  prediction?: PredictionPayload;
  smoothed_label?: string | null;
  state_machine: 'IDLE' | 'CANDIDATE' | 'ACTIVE' | 'ENDING';
  sign_sequence: string[];
  translation?: TranslationPayload;
  stage_latencies_ms: Record<string, number>;
  fps: number;
  is_valid_input: boolean;
  message?: string;
}

export interface FinalizedPhrase {
  id: string;
  timestamp: number;
  english: string;
  hindi: string;
  tokens: string[];
  confidence: number;
}

export interface VocabularyItem {
  class_id: number;
  label: string;
  gloss: string;
  english_translation: string;
  hindi_translation: string;
}

export interface AccessibilitySettings {
  largeText: boolean;
  highContrast: boolean;
  reduceMotion: boolean;
  showLandmarks: boolean;
  mirrorPreview: boolean;
  showDebugMetrics: boolean;
}
