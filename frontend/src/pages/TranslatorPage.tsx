import { useState } from 'react';
import type { AccessibilitySettings, VocabularyItem } from '../types/realtime';
import type { useCamera } from '../hooks/useCamera';
import type { useCameraDevices } from '../hooks/useCameraDevices';
import type { useLocalRealtime } from '../hooks/useLocalRealtime';
import { LOCAL_INFERENCE } from '../services/api';
import { describeSign } from '../lib/recognition';
import { Toolbar } from '../components/translator/Toolbar';
import { CameraPanel } from '../components/translator/CameraPanel';
import { DetectedSign } from '../components/translator/DetectedSign';
import { TopPredictions } from '../components/translator/TopPredictions';
import { TranslationCard } from '../components/translator/TranslationCard';
import { TranscriptCard } from '../components/translator/TranscriptCard';
import { StatsRow } from '../components/translator/StatsRow';
import { TelemetryPanel } from '../components/translator/TelemetryPanel';

interface Props {
  settings: AccessibilitySettings;
  onSettingsChange: (updater: (s: AccessibilitySettings) => AccessibilitySettings) => void;
  camera: ReturnType<typeof useCamera>;
  devices: ReturnType<typeof useCameraDevices>;
  /** useLocalRealtime and useRealtime return the same shape */
  recognition: ReturnType<typeof useLocalRealtime>;
  vocab: Map<string, VocabularyItem>;
  classes: number;
  onOpenVocabulary: () => void;
  onOpenPrivacy: () => void;
}

export function TranslatorPage({ settings, onSettingsChange, camera, devices, recognition, vocab, classes, onOpenVocabulary, onOpenPrivacy }: Props) {
  const { latestMessage: msg, transcript, fps, connectionStatus, pauseStreaming, resumeStreaming, resetSession } = recognition;
  const view = describeSign(msg);

  // The classifier only runs when a sign ends, so keep its last timing on screen
  const [inferenceMs, setInferenceMs] = useState<number | null>(null);
  const stgcnMs = msg?.stage_latencies_ms?.stgcn_inference;
  if (stgcnMs !== undefined && stgcnMs !== inferenceMs) setInferenceMs(stgcnMs);

  const start = () => camera.startCamera(devices.selectedId);
  const pause = () => {
    camera.pauseCamera();
    pauseStreaming();
  };
  const resume = () => {
    camera.resumeCamera();
    resumeStreaming();
  };
  const selectDevice = (id: string) => {
    devices.select(id);
    // Switch right away when the camera is already on
    if (camera.status === 'running' || camera.status === 'paused') camera.startCamera(id || undefined);
  };

  const cameraOn = camera.status === 'running' || camera.status === 'paused';

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 flex flex-col gap-5">
      <h1 className="sr-only">Live Translator</h1>
      <Toolbar settings={settings} onChange={onSettingsChange} onOpenVocabulary={onOpenVocabulary} onOpenPrivacy={onOpenPrivacy} />

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        <div className="lg:col-span-7 flex flex-col gap-5">
          <CameraPanel
            videoRef={camera.videoRef}
            cameraStatus={camera.status}
            cameraError={camera.errorMessage}
            modelStatus={connectionStatus}
            signing={view.state === 'signing'}
            mirror={settings.mirrorPreview}
            devices={devices.devices}
            selectedDeviceId={devices.selectedId}
            onSelectDevice={selectDevice}
            onStart={start}
            onPause={pause}
            onResume={resume}
            onStop={camera.stopCamera}
            onReset={resetSession}
            onToggleMirror={() => onSettingsChange((s) => ({ ...s, mirrorPreview: !s.mirrorPreview }))}
          />
          <StatsRow
            inferenceMs={inferenceMs}
            fps={fps}
            running={camera.status === 'running'}
            classes={classes}
            modelStatus={connectionStatus}
            local={LOCAL_INFERENCE}
          />
          {settings.showDebugMetrics && <TelemetryPanel message={msg} fps={fps} />}
        </div>

        <div className="lg:col-span-5 flex flex-col gap-5">
          <DetectedSign view={view} vocab={vocab} cameraOn={cameraOn} largeText={settings.largeText} />
          <TopPredictions prediction={msg?.prediction} vocab={vocab} />
          <TranslationCard
            translation={msg?.translation}
            sequence={msg?.sign_sequence ?? []}
            vocab={vocab}
            largeText={settings.largeText}
            onClear={resetSession}
          />
          <TranscriptCard transcript={transcript} onClear={resetSession} largeText={settings.largeText} />
        </div>
      </div>
    </div>
  );
}
