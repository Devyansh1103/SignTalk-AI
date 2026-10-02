# ADR-002: Adoption of FastAPI as Modular Monolith Backend

**Status:** APPROVED  
**Date:** October 2026  
**Context:** The backend must handle high-throughput WebSocket streams ($\ge 25\text{ FPS}$ per client), interface directly with PyTorch scientific tensors, manage sliding buffers, and expose RESTful configuration endpoints with minimal overhead.

## Decision
Adopt **Python 3.11** with **FastAPI** as the modular monolith backend framework.

## Evaluated Alternatives
1. **Flask:** Simple, but lacks native asynchronous event loop (`asyncio`) support and native WebSocket primitives without heavy external wrappers (Flask-SocketIO).
2. **Django / Django Channels:** Excessive architectural overhead, ORM complexity, and database coupling unnecessary for an in-memory streaming inference server.
3. **Node.js / Express:** Strong async I/O, but lacks native Python scientific computing and deep learning ecosystem (PyTorch, NumPy, MediaPipe).

## Consequences
- **Positive:** Native async WebSocket support running on Starlette / Uvicorn; automated OpenAPI documentation; direct in-process PyTorch model evaluation without inter-process IPC overhead.
- **Negative:** Python Global Interpreter Lock (GIL) requires running compute-heavy neural network inference inside thread-pool executors (`run_in_executor`).
