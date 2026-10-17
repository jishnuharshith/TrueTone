"""tests/test_mock_service.py

Focused unit and integration tests for FastAPI backend mock checkpoint:
1. Safe public health check (no environment/architecture/secrets leakage).
2. Contract examples allowlist and path-traversal rejection.
3. Authenticated WebSocket /v1/app/call-events:
   - Rejection of tokens passed via query string (query tokens prohibited).
   - Rejection when Authorization header is missing or invalid.
   - Fail-closed when server APP_AUTH_TOKEN is unset.
   - Successful streaming when valid token is supplied via Authorization: Bearer <token>.
   - Verification that ping and trigger_call controls are inaccessible to unauthenticated clients.
4. ASGI/Uvicorn handshake semantics verification:
   - TestClient observes WebSocketDisconnect(code=1008).
   - Live Uvicorn network clients observe HTTP 403 handshake rejection prior to upgrade.
"""

import asyncio
import threading
import time
import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient
from pydantic import TypeAdapter
import uvicorn
import websockets

from backend.config import settings
from backend.main import ALLOWED_FIXTURE_NAMES, app
from contracts.models import (
    AppEvent,
    CallConnectedEvent,
    CallEndedEvent,
    CallRingingEvent,
    CallRiskUpdateEvent,
)

client = TestClient(app)

TEST_AUTH_TOKEN = "test-secret-token-xyz-123"


# ============================================================================
# 1. Safe Health Endpoint
# ============================================================================

def test_safe_public_health_endpoint():
    """Verify GET /health returns only safe public status and no internal metadata or secrets."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data == {
        "status": "ok",
        "service": "ibm-z-datathon-backend",
        "version": "1.0.0",
    }
    # Ensure sensitive/internal fields are NOT present
    for forbidden in ["architecture", "environment", "mock_mode", "twilio", "token", "key"]:
        assert forbidden not in data, f"Health endpoint leaked internal/config field: {forbidden}"


# ============================================================================
# 2. Contract Examples Allowlist & Path Traversal Protection
# ============================================================================

def test_list_contract_examples_matches_allowlist():
    """Verify GET /v1/contracts/examples returns exactly the allowlisted fixtures."""
    response = client.get("/v1/contracts/examples")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == len(ALLOWED_FIXTURE_NAMES)
    assert set(data["examples"]) == ALLOWED_FIXTURE_NAMES


def test_get_allowlisted_contract_example_success():
    """Verify GET /v1/contracts/examples/{filename} retrieves approved fixtures."""
    response = client.get("/v1/contracts/examples/app_event_ringing.json")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "call.ringing"
    assert data["schema_version"] == "1"


def test_get_contract_example_rejects_non_allowlisted_and_traversal():
    """Verify path traversal and non-allowlisted filenames return 404."""
    # Non-allowlisted file
    assert client.get("/v1/contracts/examples/random_file.json").status_code == 404
    assert client.get("/v1/contracts/examples/config.py").status_code == 404
    assert client.get("/v1/contracts/examples/.env").status_code == 404

    # Traversal attempts
    assert client.get("/v1/contracts/examples/../main.py").status_code == 404
    assert client.get("/v1/contracts/examples/..%2fmain.py").status_code == 404
    assert client.get("/v1/contracts/examples/..%2f..%2fREADME.md").status_code == 404


# ============================================================================
# 3. Authenticated WebSocket /v1/app/call-events (TestClient Semantics)
# ============================================================================

def test_websocket_rejects_query_string_token(monkeypatch):
    """Tokens in query string MUST be rejected. Only Authorization Bearer header is accepted."""
    monkeypatch.setattr(settings, "app_auth_token", TEST_AUTH_TOKEN)

    # Attempting to supply valid token via ?token=... without Authorization header must fail
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(f"/v1/app/call-events?token={TEST_AUTH_TOKEN}") as ws:
            ws.receive_json()
    assert exc_info.value.code == 1008


def test_websocket_fails_closed_when_server_token_unset(monkeypatch):
    """When APP_AUTH_TOKEN is unset/empty on server, connection must be rejected with code 1008."""
    monkeypatch.setattr(settings, "app_auth_token", "")

    headers = {"Authorization": f"Bearer {TEST_AUTH_TOKEN}"}
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/v1/app/call-events", headers=headers) as ws:
            ws.receive_json()
    assert exc_info.value.code == 1008


def test_websocket_rejects_missing_authorization_header(monkeypatch):
    """Connecting without an Authorization header must fail with code 1008."""
    monkeypatch.setattr(settings, "app_auth_token", TEST_AUTH_TOKEN)

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/v1/app/call-events") as ws:
            ws.receive_json()
    assert exc_info.value.code == 1008


def test_websocket_rejects_invalid_bearer_token(monkeypatch):
    """Supplying an incorrect Bearer token must fail with code 1008."""
    monkeypatch.setattr(settings, "app_auth_token", TEST_AUTH_TOKEN)

    headers = {"Authorization": "Bearer wrong-token-value"}
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/v1/app/call-events", headers=headers) as ws:
            ws.receive_json()
    assert exc_info.value.code == 1008


def test_websocket_succeeds_with_valid_bearer_header(monkeypatch):
    """When client supplies valid token via Authorization header, connection succeeds and streams mock fixtures."""
    monkeypatch.setattr(settings, "app_auth_token", TEST_AUTH_TOKEN)

    headers = {"Authorization": f"Bearer {TEST_AUTH_TOKEN}"}
    with client.websocket_connect(
        "/v1/app/call-events?interval_s=0.01",
        headers=headers,
    ) as websocket:
        # Step 1: call.ringing
        event1 = websocket.receive_json()
        validated1 = TypeAdapter(AppEvent).validate_python(event1)
        assert isinstance(validated1, CallRingingEvent)

        # Step 2: call.connected
        event2 = websocket.receive_json()
        validated2 = TypeAdapter(AppEvent).validate_python(event2)
        assert isinstance(validated2, CallConnectedEvent)

        # Step 3: call.risk.update (caution)
        event3 = websocket.receive_json()
        validated3 = TypeAdapter(AppEvent).validate_python(event3)
        assert isinstance(validated3, CallRiskUpdateEvent)
        assert validated3.risk_level == "caution"

        # Step 4: call.risk.update (high)
        event4 = websocket.receive_json()
        validated4 = TypeAdapter(AppEvent).validate_python(event4)
        assert isinstance(validated4, CallRiskUpdateEvent)
        assert validated4.risk_level == "high"

        # Step 5: call.ended
        event5 = websocket.receive_json()
        validated5 = TypeAdapter(AppEvent).validate_python(event5)
        assert isinstance(validated5, CallEndedEvent)


def test_ping_and_trigger_controls_require_authentication(monkeypatch):
    """Verify ping and trigger_call are strictly inaccessible to unauthenticated callers."""
    monkeypatch.setattr(settings, "app_auth_token", TEST_AUTH_TOKEN)

    # 1. Unauthenticated client cannot ping or trigger anything (connection rejected)
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/v1/app/call-events") as ws:
            ws.send_json({"action": "ping"})
            ws.receive_json()
    assert exc_info.value.code == 1008

    # 2. Authenticated client CAN ping and trigger_call
    headers = {"Authorization": f"Bearer {TEST_AUTH_TOKEN}"}
    with client.websocket_connect(
        "/v1/app/call-events?simulate=false",
        headers=headers,
    ) as ws:
        ws.send_json({"action": "ping"})
        reply = ws.receive_json()
        assert reply["type"] == "pong"

        ws.send_json({"action": "trigger_call"})
        first_event = ws.receive_json()
        assert first_event["type"] == "call.ringing"


# ============================================================================
# 4. Live ASGI/Uvicorn Network Handshake Semantics Verification
# ============================================================================

def test_live_uvicorn_handshake_rejection_semantics(monkeypatch):
    """Verify live Uvicorn ASGI handshake semantics:

    When an unauthenticated network client initiates WebSocket handshake:
    - Because websocket.close(code=1008) is invoked BEFORE websocket.accept(),
      Uvicorn refuses the upgrade and returns an HTTP 403 Forbidden handshake response.
    - An authenticated client with Authorization: Bearer <token> header succeeds.
    """
    monkeypatch.setattr(settings, "app_auth_token", TEST_AUTH_TOKEN)

    # Launch live Uvicorn on an ephemeral port in background thread
    server_port = 8998
    config = uvicorn.Config(app=app, host="127.0.0.1", port=server_port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait briefly for server to bind
    for _ in range(20):
        if server.started:
            break
        time.sleep(0.05)

    async def run_live_client_checks():
        # Case A: Unauthenticated client (no Authorization header)
        # Observes HTTP 403 handshake denial (InvalidStatus)
        with pytest.raises(websockets.exceptions.InvalidStatus) as exc_info:
            async with websockets.connect(f"ws://127.0.0.1:{server_port}/v1/app/call-events"):
                pass
        # Assert client observes HTTP 403 rejection during handshake
        assert exc_info.value.response.status_code == 403

        # Case B: Authenticated client (valid Authorization: Bearer header)
        # Upgrades cleanly to WebSocket and receives first event
        auth_headers = [("Authorization", f"Bearer {TEST_AUTH_TOKEN}")]
        async with websockets.connect(
            f"ws://127.0.0.1:{server_port}/v1/app/call-events?interval_s=0.01",
            additional_headers=auth_headers,
        ) as ws:
            first_msg = await ws.recv()
            assert "call.ringing" in first_msg

    try:
        asyncio.run(run_live_client_checks())
    finally:
        server.should_exit = True
        thread.join(timeout=2)
