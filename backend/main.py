"""backend/main.py

FastAPI backend entrypoint for IBM Z Datathon 2026 prototype.

Provides:
- GET /health : Safe public health check (no environment/architecture/secrets).
- GET /v1/contracts/examples : Index of allowed static contract JSON fixtures.
- GET /v1/contracts/examples/{filename} : Specific allowed contract fixture (allowlisted, path-traversal protected).
- WS /v1/app/call-events : Authenticated deterministic mock app-event WebSocket delivering
  contract fixtures to Android clients. Operates strictly in mock mode using
  predefined fixtures with zero real caller audio or external credentials.
  Fails closed if APP_AUTH_TOKEN is unset or invalid. Never logs tokens.

Twilio voice webhook and Media Streams routes use signature validation and a
clearly labeled no-inference mock. Audio and transcripts are never persisted.
"""

import asyncio
import base64
import binascii
import hmac
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from fastapi import (
    FastAPI,
    HTTPException,
    Header,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import TypeAdapter
from pydantic import BaseModel, Field
from starlette.datastructures import QueryParams
from starlette.responses import Response
from twilio.request_validator import RequestValidator

from backend.config import settings
from behavior_ml.text import classify_text
from backend.live_audio import CallerAudioWindow, assess_caller_window
from behavior_ml.asr import analyze_window_with_asr
from behavior_ml.models import BehaviorWindow
from backend.live_audio import LinuxOneWhisper
from backend.line_type import lookup_line_type
from contracts.models import AppEvent, BehaviorResult, RiskLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")

app = FastAPI(
    title="IBM Z Datathon 2026 - Real-Time AI Backend (Mock Checkpoint)",
    description=(
        "FastAPI service running on IBM LinuxONE s390x. "
        "Provides authenticated deterministic mock events and static contract examples for Android integration."
    ),
    version=settings.version,
)

_latest_live_result: Dict[str, Any] = {"status": "waiting_for_call"}
_mic_in_use = False

# Enable CORS for local Android emulator / web testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_TWILIO_WEBHOOK_BYTES = 64 * 1024
MAX_TWILIO_WS_MESSAGE_CHARS = 2 * 1024 * 1024
MAX_TWILIO_MEDIA_BYTES = 1024 * 1024
TWILIO_MEDIA_PATH = "/v1/twilio/media"

CONTRACTS_DIR = (Path(__file__).resolve().parent.parent / "contracts" / "examples").resolve()

# Strict allowlist of approved contract fixtures
ALLOWED_FIXTURE_NAMES = frozenset({
    "app_event_ringing.json",
    "app_event_connected.json",
    "app_event_risk_update_caution.json",
    "app_event_risk_update_high.json",
    "app_event_ended.json",
    "app_event_error.json",
    "behavior_result_low.json",
    "behavior_result_high.json",
    "voice_token_response.json",
})


# ============================================================================
# Safe Health & Status Endpoint (Hardened for Public Exposure)
# ============================================================================

@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, str]:
    """Safe public health check endpoint.

    Returns minimal status indicator without exposing internal architecture,
    environment flags, configuration values, or secrets.
    """
    return {
        "status": "ok",
        "service": "ibm-z-datathon-backend",
        "version": settings.version,
    }


class ScriptedDemoInput(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    language: str = "auto"


@app.post("/v1/demo/analyze-scripted-text", tags=["Demo"])
async def analyze_scripted_text(payload: ScriptedDemoInput) -> Dict[str, Any]:
    """Offline prototype demo using user-supplied scripted text; no call audio."""
    if not settings.mock_mode:
        raise HTTPException(status_code=503, detail="Scripted demo is disabled")
    if payload.language not in {"auto", "en", "hi"}:
        raise HTTPException(status_code=422, detail="Invalid language")
    result = classify_text(payload.text, payload.language)
    return {
        "language": result.language,
        "risk_level": result.risk_level,
        "confidence": result.confidence,
        "evidence_labels": result.evidence_labels,
        "model_version": result.model_version,
        "source": "scripted_text_demo",
    }


@app.get("/demo", tags=["Demo"])
async def demo_page() -> Response:
    if not settings.mock_mode:
        raise HTTPException(status_code=404, detail="Unavailable")
    page = Path(__file__).with_name("demo.html").read_text(encoding="utf-8")
    page = page.replace("</html>", '<script src="/demo/microphone.js?v=4"></script></html>')
    return Response(content=page, media_type="text/html", headers={"Cache-Control": "no-store"})


@app.get("/demo/microphone.js", tags=["Demo"])
async def demo_microphone_script() -> Response:
    if not settings.mock_mode:
        raise HTTPException(status_code=404, detail="Unavailable")
    return Response(content=Path(__file__).with_name("microphone.js").read_text(encoding="utf-8"),
                    media_type="application/javascript", headers={"Cache-Control": "no-store"})


@app.get("/v1/demo/live-result", tags=["Demo"])
async def latest_live_result() -> Dict[str, Any]:
    if not settings.mock_mode:
        raise HTTPException(status_code=404, detail="Unavailable")
    return dict(_latest_live_result)


@app.websocket("/v1/demo/microphone")
async def demo_microphone(websocket: WebSocket) -> None:
    """Short consented browser microphone demo; 16 kHz PCM, memory only."""
    global _mic_in_use
    if not settings.mock_mode or not settings.live_audio_demo or _mic_in_use:
        await websocket.close(code=1008)
        return
    _mic_in_use = True
    await websocket.accept()
    session_id = str(uuid.uuid4())
    audio = bytearray()
    pending: Optional[asyncio.Task] = None
    queued: Optional[tuple[int, bytes]] = None

    def start_next(_finished: asyncio.Task) -> None:
        nonlocal pending, queued
        if (_latest_live_result.get("status") == "analyzed"
                and _latest_live_result.get("risk_level") == "high"):
            queued = None
            pending = None
            return
        if queued is None:
            pending = None
            return
        next_sequence, next_window = queued
        queued = None
        _latest_live_result.clear()
        _latest_live_result.update({"status": "processing", "source": "browser_microphone"})
        pending = asyncio.create_task(_process_pcm_window(session_id, next_sequence, next_window))
        pending.add_done_callback(start_next)
    sequence = 0
    bytes_seen = 0
    _latest_live_result.clear()
    _latest_live_result.update({"status": "listening", "source": "browser_microphone"})
    try:
        while bytes_seen < 1_600_000:  # At most 50 seconds of 16 kHz PCM.
            chunk = await asyncio.wait_for(websocket.receive_bytes(), timeout=30)
            if len(chunk) > 32768 or len(chunk) % 2:
                await websocket.close(code=1008)
                return
            bytes_seen += len(chunk)
            audio.extend(chunk)
            if len(audio) >= 128000:
                window = bytes(audio[:128000])
                # Keep two seconds of overlap so phrases near a boundary
                # appear intact in the next four-second ASR window.
                del audio[:64000]
                if pending is None:
                    if _latest_live_result.get("status") != "analyzed" or _latest_live_result.get("risk_level") != "high":
                        _latest_live_result.clear()
                        _latest_live_result.update({"status": "processing", "source": "browser_microphone"})
                    pending = asyncio.create_task(_process_pcm_window(session_id, sequence, window))
                    pending.add_done_callback(start_next)
                else:
                    # Keep the most recent full window while Whisper is busy.
                    # It is still analyzed after the browser stops recording.
                    queued = (sequence, window)
                sequence += 1
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    finally:
        audio.clear()
        _mic_in_use = False


async def _process_pcm_window(session_id: str, sequence: int, pcm: bytes) -> None:
    try:
        request = BehaviorWindow("1", session_id, sequence, "hi", 16000, pcm,
                                 sequence * 2000, sequence * 2000 + 4000)
        result = await asyncio.to_thread(analyze_window_with_asr, request, LinuxOneWhisper())
        if (_latest_live_result.get("status") == "analyzed"
                and _latest_live_result.get("risk_level") == "high"
                and result.risk_level != "high"):
            return
        _latest_live_result.clear()
        _latest_live_result.update({"status": "analyzed", "source": "browser_microphone",
                                    "risk_level": result.risk_level,
                                    "confidence": result.confidence,
                                    "evidence_labels": result.evidence_labels,
                                    "language": result.language, "sequence": sequence,
                                    "model_version": result.model_version})
    except Exception:
        _latest_live_result.clear()
        _latest_live_result.update({"status": "analysis_unavailable"})


async def _process_live_window(session_id: str, sequence: int, window: bytes,
                               line_type: str = "unknown") -> None:
    try:
        result = await asyncio.to_thread(assess_caller_window, session_id, sequence, window)
        _latest_live_result.clear()
        _latest_live_result.update({"status": "analyzed", "risk_level": result.risk_level,
                                    "source": "twilio_media", "line_type": line_type,
                                    "confidence": result.confidence,
                                    "evidence_labels": result.evidence_labels,
                                    "language": result.language, "sequence": sequence,
                                    "model_version": result.model_version})
    except Exception:
        logger.warning("Live audio window could not be analyzed")
        _latest_live_result.clear()
        _latest_live_result.update({"status": "analysis_unavailable"})


def _canonical_public_origin() -> str:
    """Return configured public HTTPS origin; never trust Host/forwarded headers."""
    value = settings.public_base_url.strip().rstrip("/")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise HTTPException(status_code=503, detail="Twilio public URL is not configured")
    return value


def _twilio_validator() -> RequestValidator:
    if not settings.twilio_auth_token:
        raise HTTPException(status_code=503, detail="Twilio validation is not configured")
    return RequestValidator(settings.twilio_auth_token)


def _signature_valid(urls: List[str], params: Any, signature: Optional[str]) -> bool:
    if not signature:
        return False
    validator = _twilio_validator()
    return any(validator.validate(url, params, signature) for url in urls)


@app.post("/v1/twilio/voice", tags=["Twilio"])
async def twilio_voice_webhook(request: Request) -> Response:
    """Validate Twilio's form webhook and return a short mocked-demo TwiML call."""
    if not settings.mock_mode:
        raise HTTPException(status_code=503, detail="Live inference is not configured")
    body = await request.body()
    if len(body) > MAX_TWILIO_WEBHOOK_BYTES:
        raise HTTPException(status_code=413, detail="Webhook body too large")
    try:
        form = QueryParams(body.decode("utf-8"))
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Malformed webhook form")

    origin = _canonical_public_origin()
    public_url = f"{origin}{request.url.path}"
    if request.url.query:
        public_url = f"{public_url}?{request.url.query}"
    if not _signature_valid([public_url], form, request.headers.get("x-twilio-signature")):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    session_id = str(uuid.uuid4())
    line_type = "unknown"
    if settings.line_type_lookup:
        if settings.twilio_api_key and settings.twilio_api_secret:
            lookup_user, lookup_password = settings.twilio_api_key, settings.twilio_api_secret
        else:
            lookup_user, lookup_password = settings.twilio_account_sid, settings.twilio_auth_token
        line_type = await lookup_line_type(
            form.get("From", ""), lookup_user, lookup_password,
        )
    stream_url = f"wss://{urlsplit(origin).netloc}{TWILIO_MEDIA_PATH}"
    # Build XML with ElementTree so URL and attributes are escaped safely.
    import xml.etree.ElementTree as ET

    root = ET.Element("Response")
    start = ET.SubElement(root, "Start")
    stream = ET.SubElement(start, "Stream", {"url": stream_url, "track": "inbound_track"})
    ET.SubElement(stream, "Parameter", {"name": "sessionId", "value": session_id})
    ET.SubElement(stream, "Parameter", {"name": "lineType", "value": line_type})
    ET.SubElement(root, "Say").text = (
        "This is a controlled demonstration call. Speech analysis is experimental."
        if settings.live_audio_demo else
        "This is a mock demonstration call. No real scam analysis is running."
    )
    if settings.twilio_forward_to:
        destination = settings.twilio_forward_to.strip()
        if not destination.startswith("+") or not destination[1:].isdigit():
            raise HTTPException(status_code=503, detail="Forwarding destination is invalid")
        dial = ET.SubElement(root, "Dial", {"answerOnBridge": "true", "timeout": "25"})
        ET.SubElement(dial, "Number").text = destination
    else:
        ET.SubElement(root, "Say").text = "Call forwarding is not configured."
        ET.SubElement(root, "Pause", {"length": "60"})
    xml_body = ET.tostring(root, encoding="unicode", short_empty_elements=True)
    return Response(content=xml_body, media_type="application/xml")


@app.post("/v1/twilio/trial-voice", tags=["Twilio"])
async def twilio_trial_voice(request: Request) -> Response:
    """Trial-compatible phone demo using Twilio speech recognition.

    Twilio trial accounts strip Media Streams, so this route collects one
    spoken phrase with Gather and passes only Twilio's text to the prototype.
    """
    if not settings.mock_mode:
        raise HTTPException(status_code=503, detail="Trial demo is disabled")
    _latest_live_result.clear()
    _latest_live_result.update({"status": "trial_webhook_received"})
    body = await request.body()
    if len(body) > MAX_TWILIO_WEBHOOK_BYTES:
        raise HTTPException(status_code=413, detail="Webhook body too large")
    try:
        form = QueryParams(body.decode("utf-8"))
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Malformed webhook form")
    origin = _canonical_public_origin()
    public_url = f"{origin}{request.url.path}"
    if request.url.query:
        public_url += f"?{request.url.query}"
    if not _signature_valid([public_url], form, request.headers.get("x-twilio-signature")):
        _latest_live_result.clear()
        _latest_live_result.update({
            "status": "trial_signature_rejected",
            "signature_present": bool(request.headers.get("x-twilio-signature")),
        })
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    _latest_live_result.clear()
    _latest_live_result.update({"status": "trial_waiting_for_speech"})

    import xml.etree.ElementTree as ET

    root = ET.Element("Response")
    gather = ET.SubElement(root, "Gather", {
        "input": "speech", "language": "hi-IN", "speechTimeout": "auto",
        "timeout": "8", "method": "POST",
        "action": f"{origin}/v1/twilio/trial-speech",
    })
    ET.SubElement(gather, "Say", {"language": "en-US"}).text = (
        "TrueTone trial demo. After the tone, speak one test sentence."
    )
    ET.SubElement(root, "Say").text = "No speech was received. Goodbye."
    return Response(content=ET.tostring(root, encoding="unicode"), media_type="application/xml")


@app.post("/v1/twilio/trial-speech", tags=["Twilio"])
async def twilio_trial_speech(request: Request) -> Response:
    """Assess one Twilio-recognized utterance without storing a transcript."""
    if not settings.mock_mode:
        raise HTTPException(status_code=503, detail="Trial demo is disabled")
    _latest_live_result.clear()
    _latest_live_result.update({"status": "trial_speech_callback_received"})
    body = await request.body()
    if len(body) > MAX_TWILIO_WEBHOOK_BYTES:
        raise HTTPException(status_code=413, detail="Webhook body too large")
    try:
        form = QueryParams(body.decode("utf-8"))
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Malformed webhook form")
    origin = _canonical_public_origin()
    public_url = f"{origin}{request.url.path}"
    if request.url.query:
        public_url += f"?{request.url.query}"
    if not _signature_valid([public_url], form, request.headers.get("x-twilio-signature")):
        _latest_live_result.clear()
        _latest_live_result.update({
            "status": "trial_speech_signature_rejected",
            "signature_present": bool(request.headers.get("x-twilio-signature")),
        })
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    phrase = form.get("SpeechResult", "").strip()[:500]
    _latest_live_result.clear()
    if phrase:
        result = classify_text(phrase, "auto")
        _latest_live_result.update({
            "status": "analyzed", "source": "twilio_gather",
            "risk_level": result.risk_level, "confidence": result.confidence,
            "evidence_labels": result.evidence_labels, "language": result.language,
            "model_version": "twilio-speech+" + result.model_version,
        })
    else:
        _latest_live_result.update({
            "status": "analyzed", "source": "twilio_gather",
            "risk_level": "insufficient_evidence", "confidence": 0.0,
            "evidence_labels": [], "language": "unknown",
            "model_version": "twilio-speech+text-prototype-v1",
        })
    import xml.etree.ElementTree as ET

    root = ET.Element("Response")
    ET.SubElement(root, "Say").text = "Thank you. Check the TrueTone demo screen for the result."
    return Response(content=ET.tostring(root, encoding="unicode"), media_type="application/xml")


def mock_analyze_media(session_id: str, sequence: int, audio_payload: bytes) -> BehaviorResult:
    """Return an explicitly mocked result; supplied audio is neither kept nor analyzed."""
    del audio_payload
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


def _media_signature_urls() -> List[str]:
    """Pin validation to the configured host/path and documented URL variants."""
    origin = _canonical_public_origin()
    host = urlsplit(origin).netloc
    https_url = f"https://{host}{TWILIO_MEDIA_PATH}"
    wss_url = f"wss://{host}{TWILIO_MEDIA_PATH}"
    # Twilio recommends testing the trailing slash variant for Voice WSS.
    # Scheme handling is restricted to these trusted configured-host candidates.
    return [https_url, https_url + "/", wss_url, wss_url + "/"]


@app.websocket(TWILIO_MEDIA_PATH)
async def twilio_media_websocket(websocket: WebSocket) -> None:
    """Receive Twilio Media Streams in memory and create mock assessments."""
    if not settings.mock_mode or not settings.twilio_auth_token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unavailable")
        return
    if websocket.url.query:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Query not allowed")
        return
    try:
        valid_signature = _signature_valid(
            _media_signature_urls(), {}, websocket.headers.get("x-twilio-signature")
        )
    except HTTPException:
        valid_signature = False
    if not valid_signature:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized")
        return

    await websocket.accept()
    session_id: Optional[str] = None
    stream_sid: Optional[str] = None
    started = False
    mock_emitted = False
    sequence = 0
    live_window = CallerAudioWindow()
    live_task: Optional[asyncio.Task] = None
    line_type = "unknown"
    try:
        while True:
            raw_message = await websocket.receive_text()
            if len(raw_message) > MAX_TWILIO_WS_MESSAGE_CHARS:
                await websocket.close(code=1009, reason="Message too large")
                return
            try:
                message = json.loads(raw_message)
            except (json.JSONDecodeError, TypeError):
                await websocket.close(code=1007, reason="Malformed message")
                return
            if not isinstance(message, dict):
                await websocket.close(code=1008, reason="Invalid event")
                return

            event = message.get("event")
            if event == "connected":
                continue
            if event == "start":
                start = message.get("start")
                if started or not isinstance(start, dict):
                    await websocket.close(code=1008, reason="Invalid start event")
                    return
                stream_sid = start.get("streamSid")
                custom = start.get("customParameters") or {}
                candidate = custom.get("sessionId") if isinstance(custom, dict) else None
                candidate_line_type = custom.get("lineType") if isinstance(custom, dict) else None
                if not isinstance(stream_sid, str) or not stream_sid or not isinstance(candidate, str):
                    await websocket.close(code=1008, reason="Invalid start event")
                    return
                session_id = candidate
                line_type = candidate_line_type if isinstance(candidate_line_type, str) and candidate_line_type in {
                    "mobile", "landline", "fixedVoip", "nonFixedVoip", "personal",
                    "tollFree", "premium", "sharedCost", "uan", "voicemail", "pager", "unknown"
                } else "unknown"
                started = True
                if settings.live_audio_demo:
                    _latest_live_result.clear()
                    _latest_live_result.update({"status": "listening", "source": "twilio_media",
                                                "line_type": line_type})
                continue
            if event == "media":
                media = message.get("media")
                if not started or not isinstance(media, dict):
                    await websocket.close(code=1008, reason="Invalid media event")
                    return
                payload = media.get("payload")
                if not isinstance(payload, str):
                    await websocket.close(code=1008, reason="Invalid media payload")
                    return
                try:
                    audio_payload = base64.b64decode(payload, validate=True)
                except (binascii.Error, ValueError):
                    await websocket.close(code=1007, reason="Invalid media payload")
                    return
                if len(audio_payload) > MAX_TWILIO_MEDIA_BYTES:
                    await websocket.close(code=1009, reason="Media payload too large")
                    return
                if settings.live_audio_demo:
                    if media.get("track") not in (None, "inbound"):
                        continue
                    try:
                        ready = live_window.add_payload(payload)
                    except ValueError:
                        await websocket.close(code=1008, reason="Invalid media frame")
                        return
                    if ready is not None:
                        if live_task is None or live_task.done():
                            live_task = asyncio.create_task(
                                _process_live_window(session_id or "", live_window.sequence, ready, line_type)
                            )
                        live_window.sequence += 1
                    continue
                if not mock_emitted:
                    result = mock_analyze_media(session_id or "", sequence, audio_payload)
                    # Emit one assessment per call in this mock checkpoint to
                    # avoid per-frame log volume. It is not sent as a real alert.
                    logger.info(
                        "Mock assessment generated: risk_level=%s model_version=%s",
                        result.risk_level.value,
                        result.model_version,
                    )
                    sequence += 1
                    mock_emitted = True
                    del result
                del audio_payload, payload, message, raw_message
                continue
            if event == "stop":
                if not started:
                    await websocket.close(code=1008, reason="Invalid stop event")
                    return
                await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
                break
            # Ignore documented non-media events without echoing or storing contents.
            if event in {"dtmf", "mark"}:
                continue
            await websocket.close(code=1008, reason="Unsupported event")
            return
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.warning("Twilio media WebSocket ended after an internal processing error")
        try:
            await websocket.close(code=1011, reason="Processing error")
        except Exception:
            pass
    finally:
        live_window.data.clear()
        # Per-connection identifiers/counters are local only; no call state,
        # audio, transcript, or caller metadata is persisted.
        session_id = None
        stream_sid = None
        sequence = 0


# ============================================================================
# Static Contract Examples Endpoints (Allowlisted & Path-Traversal Protected)
# ============================================================================

@app.get("/v1/contracts/examples", tags=["Contracts"])
async def list_contract_examples() -> Dict[str, Any]:
    """Lists only approved, allowlisted static contract JSON fixtures."""
    return {
        "count": len(ALLOWED_FIXTURE_NAMES),
        "examples": sorted(list(ALLOWED_FIXTURE_NAMES)),
    }


@app.get("/v1/contracts/examples/{filename}", tags=["Contracts"])
async def get_contract_example(filename: str) -> Dict[str, Any]:
    """Retrieves a specific static contract JSON fixture by filename.

    Strictly restricted to ALLOWED_FIXTURE_NAMES and verifies resolved path
    stays within the contract fixtures directory to prevent traversal attacks.
    """
    if filename not in ALLOWED_FIXTURE_NAMES:
        raise HTTPException(status_code=404, detail="Example fixture not found")

    file_path = (CONTRACTS_DIR / filename).resolve()

    # Path traversal safeguard: verify resolved file is strictly inside CONTRACTS_DIR
    try:
        file_path.relative_to(CONTRACTS_DIR)
    except ValueError:
        raise HTTPException(status_code=404, detail="Example fixture not found")

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Example fixture not found")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to read contract fixture '{filename}': {e}")
        raise HTTPException(status_code=500, detail="Failed to load fixture")


# ============================================================================
# Authenticated Deterministic Mock App-Event WebSocket
# ============================================================================

DETERMINISTIC_FIXTURE_SEQUENCE = [
    "app_event_ringing.json",
    "app_event_connected.json",
    "app_event_risk_update_caution.json",
    "app_event_risk_update_high.json",
    "app_event_ended.json",
]


def load_fixture(filename: str) -> Dict[str, Any]:
    """Loads and validates a fixture against the AppEvent Pydantic union."""
    file_path = (CONTRACTS_DIR / filename).resolve()
    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    validated = TypeAdapter(AppEvent).validate_python(raw)
    return validated.model_dump(mode="json")


def is_authenticated_token(provided_token: Optional[str]) -> bool:
    """Verifies provided token against settings.app_auth_token using constant-time comparison.

    Fails closed if settings.app_auth_token is empty or unset.
    Never logs the token value.
    """
    expected_token = settings.app_auth_token
    if not expected_token:
        # Fail closed: token is not configured on server
        logger.warning("WebSocket authentication rejected: APP_AUTH_TOKEN is not configured (failing closed)")
        return False

    if not provided_token:
        return False

    # Constant-time comparison to prevent timing attacks
    return hmac.compare_digest(provided_token.strip(), expected_token.strip())


@app.websocket("/v1/app/call-events")
async def mock_call_events_websocket(
    websocket: WebSocket,
    authorization: Optional[str] = Header(default=None, description="Bearer token header (Authorization: Bearer <token>)"),
    simulate: bool = Query(default=True, description="Automatically stream deterministic mock sequence"),
    interval_s: float = Query(default=1.5, description="Interval in seconds between mock events"),
):
    """Authenticated deterministic mock app-event WebSocket endpoint.

    Security & Access Control:
    - Requires authentication strictly via 'Authorization: Bearer <token>' header.
    - Query-string tokens are rejected to prevent leakage in proxy/access logs.
    - Configured solely via the private APP_AUTH_TOKEN environment variable (no hardcoded default).
    - Temporary mock/development credential ONLY. Not for production use.
    - Fails closed with close code 1008 (Policy Violation) / HTTP 403 handshake rejection
      if APP_AUTH_TOKEN is unset or token is invalid.
    - Never logs token values.
    - All controls (ping, trigger_call, and event stream) are accessible ONLY to authenticated clients.

    Operation:
    - Streams harmless static fixtures in deterministic order:
      call.ringing -> call.connected -> call.risk.update (caution) ->
      call.risk.update (high) -> call.ended
    - Strictly inspects no real caller content or live audio.
    """
    # Never deliver fabricated risk alerts when the service is in live mode.
    if not settings.mock_mode:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Live events unavailable")
        return

    # Reject tokens in query string; accept ONLY Authorization: Bearer <token>
    candidate_token: Optional[str] = None
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            candidate_token = parts[1]

    # Validate authentication before accepting the connection
    if not is_authenticated_token(candidate_token):
        logger.warning("WebSocket connection rejected: invalid or missing Authorization Bearer token")
        # Reject connection with Policy Violation code before accept (ASGI yields HTTP 403 handshake rejection)
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized")
        return

    await websocket.accept()
    logger.info("Authenticated Android client connected to /v1/app/call-events (Mock Mode)")

    sim_task: Optional[asyncio.Task] = None
    try:
        # Pre-load deterministic sequence from validated fixtures
        fixtures_data: List[Dict[str, Any]] = []
        for name in DETERMINISTIC_FIXTURE_SEQUENCE:
            fixtures_data.append(load_fixture(name))

        async def run_simulation():
            for event in fixtures_data:
                await websocket.send_json(event)
                logger.info(f"Mock WS dispatched event: {event.get('type')}")
                await asyncio.sleep(interval_s)

        # Launch initial simulation if requested
        if simulate:
            sim_task = asyncio.create_task(run_simulation())

        # Listen for client control messages (only reachable after successful authentication)
        while True:
            text = await websocket.receive_text()
            try:
                msg = json.loads(text)
                action = msg.get("action")
                if action == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp_ms": int(asyncio.get_event_loop().time() * 1000),
                    })
                elif action == "trigger_call":
                    if sim_task and not sim_task.done():
                        sim_task.cancel()
                    sim_task = asyncio.create_task(run_simulation())
                else:
                    await websocket.send_json({
                        "type": "ack",
                        "received_action": action,
                    })
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Malformed JSON message received from client",
                })

    except WebSocketDisconnect:
        logger.info("Android client disconnected from /v1/app/call-events")
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Error in mock WebSocket session: {e}", exc_info=True)
    finally:
        if sim_task and not sim_task.done():
            sim_task.cancel()
