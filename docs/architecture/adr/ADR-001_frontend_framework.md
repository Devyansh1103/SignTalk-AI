# ADR-001: Adoption of React and TypeScript for Frontend

**Status:** APPROVED  
**Date:** October 2026  
**Context:** SignTalk AI requires a responsive, low-friction web client capable of managing high-frequency webcam streams, rendering HTML5 canvas overlays, managing bi-directional WebSockets, and displaying high-contrast live captions conforming to accessibility standards.

## Decision
Adopt **React 18** with **TypeScript** as the official client application framework.

## Evaluated Alternatives
1. **Vanilla JavaScript / HTML5:** While lightweight, it lacks structured component reusability and type safety, leading to difficult state management across complex streaming and confidence states.
2. **Vue.js / Svelte:** Excellent reactivity, but React possesses a significantly broader ecosystem of accessible component libraries and real-time canvas tooling.
3. **Desktop Native (PyQt / Tkinter / Electron):** Heavy distribution footprint; difficult for non-technical users to install; lacks browser-based zero-install convenience.

## Consequences
- **Positive:** Type safety across WebSocket message contracts; robust component isolation (CameraView, CaptionPanel); rapid rendering via React virtual DOM.
- **Negative:** Requires Node.js build tooling (Vite) and npm dependency management.
