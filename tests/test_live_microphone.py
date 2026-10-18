"""A browser PCM stream reaches the backend assessment path."""
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

import backend.main as backend
from backend.config import settings


def test_microphone_window_reaches_analyzer(monkeypatch):
    monkeypatch.setattr(settings, "mock_mode", True)
    monkeypatch.setattr(settings, "live_audio_demo", True)

    def fake_analyze(request, transcriber):
        assert len(request.pcm_s16le) == 128000
        assert request.language_hint == "hi"
        return SimpleNamespace(risk_level="high", confidence=0.75,
                               evidence_labels=["otp_request"], language="hi",
                               model_version="test-transcriber")

    monkeypatch.setattr(backend, "analyze_window_with_asr", fake_analyze)
    with TestClient(backend.app) as client:
        with client.websocket_connect("/v1/demo/microphone") as ws:
            for _ in range(8):
                ws.send_bytes(b"\x01\x00" * 8000)
            for _ in range(100):
                result = client.get("/v1/demo/live-result").json()
                if result.get("status") == "analyzed":
                    break
                time.sleep(0.01)
            assert result["risk_level"] == "high"
            assert result["evidence_labels"] == ["otp_request"]
