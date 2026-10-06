import { useState, useEffect } from 'react';
import { useCamera } from './hooks/useCamera';
import { useRealtime } from './hooks/useRealtime';
import { fetchVocabulary } from './services/api';
import type { AccessibilitySettings, VocabularyItem } from './types/realtime';

import { Header } from './components/layout/Header';
import { Footer } from './components/layout/Footer';
import { CameraView } from './components/camera/CameraView';
import { SessionControls } from './components/controls/SessionControls';
import { AccessibilityControls } from './components/controls/AccessibilityControls';
import { CurrentSign } from './components/translation/CurrentSign';
import { TranslationPanel } from './components/translation/TranslationPanel';
import { LiveTranscript } from './components/translation/LiveTranscript';
import { DeveloperHUD } from './components/debug/DeveloperHUD';
import { VocabularyModal } from './components/modals/VocabularyModal';
import { PrivacyModal } from './components/modals/PrivacyModal';

// Certified MVP-10 Vocabulary Fallback
const DEFAULT_VOCABULARY: VocabularyItem[] = [
  { class_id: 0, label: 'hello', gloss: 'HELLO', english_translation: 'Hello', hindi_translation: 'नमस्ते' },
  { class_id: 1, label: 'thankyou', gloss: 'THANK_YOU', english_translation: 'Thank you', hindi_translation: 'धन्यवाद' },
  { class_id: 2, label: 'good', gloss: 'GOOD', english_translation: 'Good', hindi_translation: 'अच्छा' },
  { class_id: 3, label: 'happy', gloss: 'HAPPY', english_translation: 'Happy', hindi_translation: 'खुश' },
  { class_id: 4, label: 'monday', gloss: 'MONDAY', english_translation: 'Monday', hindi_translation: 'सोमवार' },
  { class_id: 5, label: 'car', gloss: 'CAR', english_translation: 'Car', hindi_translation: 'गाड़ी' },
  { class_id: 6, label: 'bird', gloss: 'BIRD', english_translation: 'Bird', hindi_translation: 'पक्षी' },
  { class_id: 7, label: 'house', gloss: 'HOUSE', english_translation: 'House', hindi_translation: 'घर' },
  { class_id: 8, label: 'time', gloss: 'TIME', english_translation: 'Time', hindi_translation: 'समय' },
  { class_id: 9, label: 'teacher', gloss: 'TEACHER', english_translation: 'Teacher', hindi_translation: 'शिक्षक' },
];

export function App() {
  // Accessibility & UX Settings
  const [settings, setSettings] = useState<AccessibilitySettings>({
    largeText: false,
    highContrast: false,
    reduceMotion: false,
    showLandmarks: true,
    mirrorPreview: true,
    showDebugMetrics: false,
  });

  // Modal State
  const [isVocabOpen, setIsVocabOpen] = useState(false);
  const [isPrivacyOpen, setIsPrivacyOpen] = useState(false);
  const [vocabulary, setVocabulary] = useState<VocabularyItem[]>(DEFAULT_VOCABULARY);

  // Camera Ingestion Hook
  const {
    videoRef,
    status: cameraStatus,
    errorMessage: cameraError,
    startCamera,
    pauseCamera,
    resumeCamera,
    stopCamera,
  } = useCamera();

  // Real-Time WebSocket Hook
  const {
    connectionStatus,
    latestMessage,
    transcript,
    fps,
    pauseStreaming,
    resumeStreaming,
    resetSession,
  } = useRealtime(videoRef, cameraStatus === 'running');

  // Load official vocabulary from backend on mount
  useEffect(() => {
    fetchVocabulary().then((items) => {
      if (items && items.length > 0) {
        setVocabulary(items);
      }
    });
  }, []);

  // Action Handlers
  const handleStart = () => {
    startCamera();
  };

  const handlePause = () => {
    pauseCamera();
    pauseStreaming();
  };

  const handleResume = () => {
    resumeCamera();
    resumeStreaming();
  };

  const handleReset = () => {
    resetSession();
  };

  const handleStop = () => {
    stopCamera();
  };

  return (
    <div
      className={`min-h-screen flex flex-col justify-between transition-colors ${
        settings.highContrast ? 'high-contrast' : ''
      } ${settings.largeText ? 'large-text' : ''}`}
    >
      {/* Top Header */}
      <Header connectionStatus={connectionStatus} />

      {/* Main Translation Interface */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 flex flex-col gap-6">
        {/* Accessibility & Settings Toolbar */}
        <AccessibilityControls
          settings={settings}
          onUpdateSettings={setSettings}
          onOpenVocabulary={() => setIsVocabOpen(true)}
          onOpenPrivacy={() => setIsPrivacyOpen(true)}
        />

        {/* Core Layout Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column: Camera View & Primary Controls (7 cols) */}
          <div className="lg:col-span-7 flex flex-col gap-4">
            <CameraView
              videoRef={videoRef}
              status={cameraStatus}
              errorMessage={cameraError}
              fps={fps}
              mirror={settings.mirrorPreview}
              onStartCamera={handleStart}
            />

            <SessionControls
              cameraStatus={cameraStatus}
              onStart={handleStart}
              onPause={handlePause}
              onResume={handleResume}
              onReset={handleReset}
              onStop={handleStop}
            />

            {settings.showDebugMetrics && (
              <DeveloperHUD message={latestMessage} fps={fps} />
            )}
          </div>

          {/* Right Column: AI Predictions, Translation & Live Transcript (5 cols) */}
          <div className="lg:col-span-5 flex flex-col gap-4">
            {/* Current Sign Recognition Card */}
            <CurrentSign
              prediction={latestMessage?.prediction}
              smoothedLabel={latestMessage?.smoothed_label}
              stateMachine={latestMessage?.state_machine || 'IDLE'}
              isLargeText={settings.largeText}
            />

            {/* Natural Language Translation Panel */}
            <TranslationPanel
              translation={latestMessage?.translation}
              signSequence={latestMessage?.sign_sequence || []}
              isLargeText={settings.largeText}
            />

            {/* Session Transcript History */}
            <LiveTranscript
              transcript={transcript}
              onClear={handleReset}
              isLargeText={settings.largeText}
            />
          </div>
        </div>
      </main>

      {/* Modals */}
      <VocabularyModal
        isOpen={isVocabOpen}
        onClose={() => setIsVocabOpen(false)}
        vocabulary={vocabulary}
      />

      <PrivacyModal
        isOpen={isPrivacyOpen}
        onClose={() => setIsPrivacyOpen(false)}
      />

      {/* Footer */}
      <Footer
        onOpenVocabulary={() => setIsVocabOpen(true)}
        onOpenPrivacy={() => setIsPrivacyOpen(true)}
      />
    </div>
  );
}

export default App;
