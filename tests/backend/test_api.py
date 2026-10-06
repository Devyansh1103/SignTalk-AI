"""
Backend REST & WebSocket API Integration Tests (Phase 5).
Validates /health, /api/status, /api/vocabulary, and WebSocket session handshake.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_root_endpoint(client):
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert "SignTalk AI" in data["service"]


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["vocabulary_size"] == 10


def test_api_status_endpoint(client):
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "platform" in data


def test_vocabulary_endpoint(client):
    res = client.get("/api/vocabulary")
    assert res.status_code == 200
    data = res.json()
    assert data["num_classes"] == 10
    assert len(data["classes"]) == 10
    labels = [c["label"] for c in data["classes"]]
    assert "hello" in labels
    assert "teacher" in labels


def test_websocket_handshake(client):
    with client.websocket_connect("/ws/realtime") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "status"
        assert "Connected" in msg["message"]

        # Send reset action
        ws.send_json({"action": "reset"})
        reset_resp = ws.receive_json()
        assert reset_resp["type"] == "status"
        assert "reset" in reset_resp["message"]
