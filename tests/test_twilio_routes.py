"""Focused tests for the signed Twilio webhook and mocked Media Streams route."""

import base64
import asyncio
import json
import socket
import threading
import time
from urllib.parse import urlencode

import httpx
import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient
from starlette.datastructures import QueryParams
from twilio.request_validator import RequestValidator
import uvicorn
import websockets

import backend.main as backend_main
from backend.config import settings
from contracts.models import BehaviorResult, RiskLevel


AUTH_TOKEN = "test-twilio-auth-token"
PUBLIC_ORIGIN = "https://test-tunnel.trycloudflare.com"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "twilio_auth_token", AUTH_TOKEN)
    monkeypatch.setattr(settings, "public_base_url", PUBLIC_ORIGIN)
    monkeypatch.setattr(settings, "mock_mode", True)
    with TestClient(backend_main.app) as test_client:
        yield test_client


def twilio_signature(url: str, params):
    return RequestValidator(AUTH_TOKEN).compute_signature(url, params)


def test_voice_webhook_requires_valid_signature(client):
    body = urlencode({"CallSid": "CA-test", "From": "+15550000000"})
    response = client.post(
        "/v1/twilio/voice",
        content=body,
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 403

    valid = twilio_signature(f"{PUBLIC_ORIGIN}/v1/twilio/voice", QueryParams(body))
    response = client.post(
        "/v1/twilio/voice",
        content=body,
        headers={
            "content-type": "application/x-www-form-urlencoded",
            "x-twilio-signature": valid,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    assert "wss://test-tunnel.trycloudflare.com/v1/twilio/media" in response.text
    assert "mock" in response.text.lower()
    assert "<Pause length=\"30\"" in response.text


def test_voice_webhook_uses_canonical_url_and_rejects_bad_signature(client):
    body = urlencode({"CallSid": "CA-test"})
    query = "source=event"
    url = f"{PUBLIC_ORIGIN}/v1/twilio/voice?{query}"
    sig = twilio_signature(url, QueryParams(body))
    response = client.post(
        f"/v1/twilio/voice?{query}",
        content=body,
        headers={"content-type": "application/x-www-form-urlencoded", "x-twilio-signature": sig},
    )
    assert response.status_code == 200

    bad = client.post(
        "/v1/twilio/voice",
        content=body,
        headers={"content-type": "application/x-www-form-urlencoded", "x-twilio-signature": "bad"},
    )
    assert bad.status_code == 403


def ws_headers(url: str):
    return {"X-Twilio-Signature": twilio_signature(url, {})}


def test_media_websocket_validates_signature_and_processes_mock_media(client, monkeypatch):
    observed = []

    def fake_mock(session_id, sequence, audio_payload):
        observed.append((session_id, sequence, len(audio_payload)))
        return BehaviorResult(
            session_id=session_id,
            sequence=sequence,
            language="unknown",
            confidence=0.0,
            risk_level=RiskLevel.INSUFFICIENT_EVIDENCE,
            evidence_labels=[],
            model_version="mock-no-audio-analysis-v1",
            inference_ms=0.0,
        )

    monkeypatch.setattr(backend_main, "mock_analyze_media", fake_mock)
    public_wss = f"wss://test-tunnel.trycloudflare.com{backend_main.TWILIO_MEDIA_PATH}"
    with client.websocket_connect(
        backend_main.TWILIO_MEDIA_PATH,
        headers=ws_headers(public_wss),
    ) as ws:
        ws.send_json({"event": "connected", "protocol": "Call", "version": "1.0.0"})
        ws.send_json({
            "event": "start",
            "start": {
                "streamSid": "MZ-test-stream",
                "callSid": "CA-test-call",
                "customParameters": {"sessionId": "opaque-test-session"},
            },
        })
        payload = base64.b64encode(b"\x00\x01\x02\x03").decode("ascii")
        ws.send_json({"event": "media", "media": {"track": "inbound", "payload": payload}})
        ws.send_json({"event": "stop", "stop": {"streamSid": "MZ-test-stream"}})
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()

    assert closed.value.code == 1000
    assert observed == [("opaque-test-session", 0, 4)]
    assert not hasattr(backend_main.app.state, "twilio_audio")


def test_media_websocket_rejects_invalid_signature(client):
    with pytest.raises(WebSocketDisconnect) as rejected:
        with client.websocket_connect(
            backend_main.TWILIO_MEDIA_PATH,
            headers={"X-Twilio-Signature": "invalid"},
        ):
            pass
    assert rejected.value.code == 1008


@pytest.mark.parametrize(
    "message,expected_code",
    [
        ("not-json", 1007),
        ('{"event":"media","media":{"payload":"!!!!"}}', 1008),
        ('{"event":"unknown"}', 1008),
    ],
)
def test_media_websocket_closes_on_malformed_or_out_of_order_messages(
    client, message, expected_code
):
    public_wss = f"wss://test-tunnel.trycloudflare.com{backend_main.TWILIO_MEDIA_PATH}"
    with client.websocket_connect(
        backend_main.TWILIO_MEDIA_PATH,
        headers=ws_headers(public_wss),
    ) as ws:
        ws.send_text(message)
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()
    assert closed.value.code == expected_code


def test_media_websocket_cleanup_on_client_disconnect(client):
    public_wss = f"wss://test-tunnel.trycloudflare.com{backend_main.TWILIO_MEDIA_PATH}"
    with client.websocket_connect(
        backend_main.TWILIO_MEDIA_PATH,
        headers=ws_headers(public_wss),
    ) as ws:
        ws.send_json({
            "event": "start",
            "start": {
                "streamSid": "MZ-test-stream",
                "customParameters": {"sessionId": "opaque-test-session"},
            },
        })
    # A fresh connection succeeding after the prior disconnect demonstrates
    # there is no stale global per-call state preventing cleanup.
    with client.websocket_connect(
        backend_main.TWILIO_MEDIA_PATH,
        headers=ws_headers(public_wss),
    ) as ws:
        ws.send_json({
            "event": "start",
            "start": {
                "streamSid": "MZ-test-stream-2",
                "customParameters": {"sessionId": "opaque-test-session-2"},
            },
        })
        ws.send_json({"event": "stop"})
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()
    assert closed.value.code == 1000


def test_media_websocket_rejects_invalid_base64_after_start(client):
    public_wss = f"wss://test-tunnel.trycloudflare.com{backend_main.TWILIO_MEDIA_PATH}"
    with client.websocket_connect(
        backend_main.TWILIO_MEDIA_PATH,
        headers=ws_headers(public_wss),
    ) as ws:
        ws.send_json({
            "event": "start",
            "start": {
                "streamSid": "MZ-test-stream",
                "customParameters": {"sessionId": "opaque-test-session"},
            },
        })
        ws.send_json({"event": "media", "media": {"payload": "!!!!"}})
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()
    assert closed.value.code == 1007


def test_mock_analyzer_returns_explicit_insufficient_evidence():
    result = backend_main.mock_analyze_media("opaque-session", 7, b"private audio")
    assert result.session_id == "opaque-session"
    assert result.sequence == 7
    assert result.risk_level == RiskLevel.INSUFFICIENT_EVIDENCE
    assert result.evidence_labels == []
    assert result.model_version == "mock-no-audio-analysis-v1"


def test_twilio_routes_over_live_uvicorn_network(client):
    # TestClient above validates ASGI behavior; this test exercises actual
    # HTTP and WebSocket handshakes over a local TCP listener as well.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(backend_main.app, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(40):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started

    async def exercise_routes():
        body = {"CallSid": "CA-local-test"}
        signature = twilio_signature(f"{PUBLIC_ORIGIN}/v1/twilio/voice", body)
        async with httpx.AsyncClient() as http:
            response = await http.post(
                f"http://127.0.0.1:{port}/v1/twilio/voice",
                data=body,
                headers={"X-Twilio-Signature": signature},
            )
        assert response.status_code == 200
        assert "/v1/twilio/media" in response.text

        public_wss = f"wss://test-tunnel.trycloudflare.com{backend_main.TWILIO_MEDIA_PATH}"
        headers = [("X-Twilio-Signature", twilio_signature(public_wss, {}))]
        async with websockets.connect(
            f"ws://127.0.0.1:{port}{backend_main.TWILIO_MEDIA_PATH}",
            additional_headers=headers,
        ) as ws:
            await ws.send(json.dumps({
                "event": "start",
                "start": {
                    "streamSid": "MZ-network-test",
                    "customParameters": {"sessionId": "opaque-network-session"},
                },
            }))
            await ws.send(json.dumps({
                "event": "media",
                "media": {"payload": base64.b64encode(b"abcd").decode("ascii")},
            }))
            await ws.send(json.dumps({"event": "stop"}))
            with pytest.raises(websockets.ConnectionClosedOK):
                await ws.recv()

    try:
        asyncio.run(exercise_routes())
    finally:
        server.should_exit = True
        thread.join(timeout=3)
