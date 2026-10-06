/**
 * SignTalk AI - Real-Time WebSocket Client
 * Phase 5: Resilient streaming client with bounded backoff and typed event dispatch.
 */

import type { ServerRealtimeMessage, ConnectionStatus } from '../types/realtime';

export type MessageListener = (msg: ServerRealtimeMessage) => void;
export type StatusListener = (status: ConnectionStatus) => void;

export class RealtimeClient {
  private ws: WebSocket | null = null;
  private url: string;
  private messageListeners: Set<MessageListener> = new Set();
  private statusListeners: Set<StatusListener> = new Set();
  private currentStatus: ConnectionStatus = 'disconnected';
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectTimer: number | null = null;
  private isExplicitlyClosed = false;

  constructor(url?: string) {
    this.url = url || (import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/realtime');
  }

  public getStatus(): ConnectionStatus {
    return this.currentStatus;
  }

  public subscribeMessage(listener: MessageListener): () => void {
    this.messageListeners.add(listener);
    return () => this.messageListeners.delete(listener);
  }

  public subscribeStatus(listener: StatusListener): () => void {
    this.statusListeners.add(listener);
    listener(this.currentStatus);
    return () => this.statusListeners.delete(listener);
  }

  private setStatus(status: ConnectionStatus) {
    this.currentStatus = status;
    this.statusListeners.forEach((l) => l(status));
  }

  public connect() {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this.isExplicitlyClosed = false;
    this.setStatus('connecting');

    try {
      this.ws = new WebSocket(this.url);
      this.ws.binaryType = 'arraybuffer';

      this.ws.onopen = () => {
        this.setStatus('connected');
        this.reconnectAttempts = 0;
      };

      this.ws.onmessage = (event) => {
        if (typeof event.data === 'string') {
          try {
            const data: ServerRealtimeMessage = JSON.parse(event.data);
            this.messageListeners.forEach((l) => l(data));
          } catch (e) {
            console.error('Failed to parse WebSocket JSON payload:', e);
          }
        }
      };

      this.ws.onerror = (err) => {
        console.error('WebSocket connection error:', err);
        this.setStatus('error');
      };

      this.ws.onclose = () => {
        this.ws = null;
        if (!this.isExplicitlyClosed) {
          this.setStatus('disconnected');
          this.scheduleReconnect();
        } else {
          this.setStatus('disconnected');
        }
      };
    } catch (e) {
      console.error('WebSocket initialization failure:', e);
      this.setStatus('error');
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect() {
    if (this.isExplicitlyClosed || this.reconnectAttempts >= this.maxReconnectAttempts) {
      return;
    }

    this.reconnectAttempts += 1;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts - 1), 8000);
    this.reconnectTimer = window.setTimeout(() => {
      this.connect();
    }, delay);
  }

  public sendFrameBytes(bytes: ArrayBuffer | Blob) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(bytes);
    }
  }

  public sendBase64Frame(base64Image: string) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'frame', image_base64: base64Image }));
    }
  }

  public sendControl(action: 'start' | 'pause' | 'resume' | 'reset' | 'stop') {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ action }));
    }
  }

  public disconnect() {
    this.isExplicitlyClosed = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.setStatus('disconnected');
  }
}
