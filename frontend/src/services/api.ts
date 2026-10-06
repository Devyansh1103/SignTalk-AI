/**
 * SignTalk AI - REST API Service
 */

import type { VocabularyItem } from '../types/realtime';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`Health check failed with status ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error('Failed to connect to backend /health:', err);
    return null;
  }
}

export async function fetchVocabulary(): Promise<VocabularyItem[]> {
  try {
    const res = await fetch(`${API_BASE}/api/vocabulary`);
    if (!res.ok) throw new Error(`Failed to fetch vocabulary: ${res.status}`);
    const data = await res.json();
    return data.classes || [];
  } catch (err) {
    console.error('Failed to fetch vocabulary:', err);
    return [];
  }
}

export async function fetchSystemStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/status`);
    if (!res.ok) throw new Error(`Failed to fetch system status: ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error('Failed to fetch system status:', err);
    return null;
  }
}
