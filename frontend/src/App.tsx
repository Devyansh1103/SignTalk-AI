import { useCallback, useEffect, useMemo, useState } from 'react';
import { useCamera } from './hooks/useCamera';
import { useCameraDevices } from './hooks/useCameraDevices';
import { useRealtime } from './hooks/useRealtime';
import { useLocalRealtime } from './hooks/useLocalRealtime';
import { fetchVocabulary, LOCAL_INFERENCE } from './services/api';
import type { AccessibilitySettings, VocabularyItem } from './types/realtime';
import { useRoute } from './lib/useRoute';
import { useTheme } from './lib/useTheme';

import { Header } from './components/layout/Header';
import { Footer } from './components/layout/Footer';
import { PrivacyModal } from './components/modals/PrivacyModal';
import { VocabularyModal } from './components/modals/VocabularyModal';
import { HomePage } from './pages/HomePage';
import { TranslatorPage } from './pages/TranslatorPage';
import { AboutPage } from './pages/AboutPage';
import { ContactPage } from './pages/ContactPage';

// Fixed per build, so the same hook runs on every render
const useRecognition = LOCAL_INFERENCE ? useLocalRealtime : useRealtime;

export function App() {
  const route = useRoute();
  const { theme, setTheme } = useTheme();

  const [settings, setSettings] = useState<AccessibilitySettings>({
    largeText: false,
    highContrast: false,
    reduceMotion: false,
    showLandmarks: true,
    mirrorPreview: true,
    showDebugMetrics: false,
  });
  const [vocabOpen, setVocabOpen] = useState(false);
  const [privacyOpen, setPrivacyOpen] = useState(false);
  const [vocabulary, setVocabulary] = useState<VocabularyItem[]>([]);

  const camera = useCamera();
  const devices = useCameraDevices(camera.status === 'running');
  // Lives here rather than on the translator page, so the models load once (and start loading on Home)
  const recognition = useRecognition(camera.videoRef, camera.status === 'running');

  useEffect(() => {
    fetchVocabulary().then((items) => {
      if (items.length) setVocabulary(items);
    });
  }, []);

  // Leaving the translator turns the camera off
  const { stopCamera, status: cameraStatus } = camera;
  useEffect(() => {
    if (route !== 'translator' && cameraStatus !== 'idle' && cameraStatus !== 'stopped') stopCamera();
  }, [route, cameraStatus, stopCamera]);

  const vocabByLabel = useMemo(() => new Map(vocabulary.map((v) => [v.label, v])), [vocabulary]);
  const openPrivacy = useCallback(() => setPrivacyOpen(true), []);

  return (
    <div className={`min-h-screen flex flex-col ${settings.highContrast ? 'high-contrast bg-bg' : ''} ${settings.largeText ? 'large-text' : ''}`}>
      <a href="#main" onClick={(e) => { e.preventDefault(); document.getElementById('main')?.focus(); }} className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:px-3 focus:py-2 focus:rounded-lg focus:bg-surface focus:text-ink">
        Skip to content
      </a>
      <Header route={route} theme={theme} onThemeChange={setTheme} status={recognition.connectionStatus} local={LOCAL_INFERENCE} />

      <main id="main" tabIndex={-1} className="flex-1 outline-none">
        {route === 'home' && <HomePage classes={vocabulary.length} />}
        {route === 'translator' && (
          <TranslatorPage
            settings={settings}
            onSettingsChange={setSettings}
            camera={camera}
            devices={devices}
            recognition={recognition}
            vocab={vocabByLabel}
            classes={vocabulary.length}
            onOpenVocabulary={() => setVocabOpen(true)}
            onOpenPrivacy={openPrivacy}
          />
        )}
        {route === 'about' && <AboutPage classes={vocabulary.length} />}
        {route === 'contact' && <ContactPage />}
      </main>

      <Footer onOpenPrivacy={openPrivacy} />

      <VocabularyModal open={vocabOpen} onClose={() => setVocabOpen(false)} vocabulary={vocabulary} />
      <PrivacyModal open={privacyOpen} onClose={() => setPrivacyOpen(false)} />
    </div>
  );
}

export default App;
