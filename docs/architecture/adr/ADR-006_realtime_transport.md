# ADR-006: Selection of WebSockets over WebRTC / HTTP for Real-Time Streaming

**Status:** APPROVED  
**Date:** October 2026  
**Context:** The system requires a low-latency communication transport between the React web client and the FastAPI backend to stream landmark coordinate arrays at $25-30\text{ FPS}$ and receive text captions.

## Decision
Adopt **Standard WebSockets (`ws://` / `wss://`)** as the primary real-time communication protocol.

## Evaluated Alternatives
1. **WebRTC Data Channels:** Extremely low UDP latency, but introduces substantial architectural overhead (STUN/TURN signaling servers, ICE candidate negotiation, SDP handshakes). Because SignTalk AI transmits lightweight skeletal coordinate arrays ($< 50\text{ KB/s}$) over local or LAN networks rather than heavy peer-to-peer raw video, the complexity of WebRTC is unjustified.
2. **HTTP Long-Polling / Chunked Streaming:** Introduces high TCP header overhead and latency penalties ($> 150\text{ ms}$ per request-response cycle), making sustained 30 FPS coordinate streaming impossible.
3. **gRPC Web:** High serialization efficiency via Protocol Buffers, but requires an Envoy proxy translation layer in browser environments, increasing deployment friction.

## Consequences
- **Positive:** Low transport latency ($< 15\text{ ms}$ on local/LAN); full-duplex continuous frame streaming; zero third-party signaling server dependencies; native browser and FastAPI support.
- **Negative:** Runs over TCP (subject to head-of-line blocking on congested lossy networks, though negligible on local/LAN setups).
