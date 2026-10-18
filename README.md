# IBM Z Datathon 2026 — Real-Time AI Prototype

## Integrated submission checkpoint

Start with [SUBMISSION.md](SUBMISSION.md) for the current setup, verification and limitations. The integrated Android app in `android/` sends scripted text to FastAPI, displays real backend responses, and opens the browser microphone demo. The backend test suite passes (28 tests), and the debug APK builds successfully.

Live Twilio speech analysis remains unverified. The trained classifier and AI-voice detector are not connected to this UI. The architecture below includes planned components; it is not a claim of completed end-to-end call protection.

The project targets scammer-behavior detection using **IBM LinuxONE (s390x)**, consented audio and advisory Android risk alerts.

---

## Monorepo Layout

```
.
├── backend/        # Python FastAPI service (Twilio webhooks, Media Streams, event push)
├── contracts/      # Versioned shared Pydantic schemas, vocabulary docs, and JSON examples
├── behavior-ml/    # Speech recognition & scammer behavior ML analyzer (Shaun)
├── android/        # Native Kotlin Jetpack Compose app (Twilio Voice SDK & live UI)
└── docs/           # Architecture diagrams, LinuxONE deployment evidence, runbooks
```

---

## Required Environment Variables (Names Only)

> **CRITICAL**: Never commit `.env` files, API keys, or private tokens to Git. Pass variables via a local `.env` file or LinuxONE systemd environment file.

### Backend / Twilio Configuration
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_API_KEY`
- `TWILIO_API_SECRET`
- `TWILIO_TWIML_APP_SID`
- `TWILIO_VOICE_CALLER_ID`
- `TWILIO_PUSH_CREDENTIAL_SID`
- `TWILIO_MEDIA_STREAM_URL`

### Service & Networking
- `BACKEND_PORT` (default: `8000`)
- `BACKEND_HOST` (default: `0.0.0.0`)
- `PUBLIC_BASE_URL` (public HTTPS URL, e.g. `https://<tunnel-domain>`)
- `PUBLIC_WSS_URL` (public WSS URL, e.g. `wss://<tunnel-domain>`)
- `APP_AUTH_TOKEN` (TEMPORARY mock/development credential ONLY; fail-closed when unset. Do not bake into Android APK or Git. Production will use short-lived dynamically issued credentials)
- `ENVIRONMENT` (`development` | `production`)
- `MOCK_MODE` (`true` | `false`)

---

## Run & Development Commands

### Local Backend Setup
```bash
# From workspace root
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On Linux/macOS: source .venv/bin/activate
pip install -r backend/requirements.txt

# Run backend in mock mode (no Twilio/ML credentials needed)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### LinuxONE VM (s390x) Execution
```bash
# Connect to LinuxONE Community Cloud VM
ssh -i <path-to-private-key> linux1@148.100.109.138

# Activate host python venv
source ~/jupyter-env/bin/activate

# Launch FastAPI backend service
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### Temporary Twilio Mock Checkpoint (Laptop)

`POST /v1/twilio/voice` validates `X-Twilio-Signature` and returns TwiML that
starts an inbound-only Media Stream at `/v1/twilio/media`. The media route
validates the WebSocket handshake signature and accepts Twilio `connected`,
`start`, `media`, and `stop` messages. This checkpoint performs **no audio
inference**: it creates one `insufficient_evidence` result labeled
`mock-no-audio-analysis-v1` per call. The result is only visible in the
backend's safe operational log; it is not yet delivered to the Android app.

For a temporary laptop tunnel, run FastAPI on `127.0.0.1:8000` and point
`cloudflared tunnel --url http://127.0.0.1:8000` at that local service. Set
`PUBLIC_BASE_URL` to the currently assigned `https://<random>.trycloudflare.com`
origin and set `TWILIO_AUTH_TOKEN` in the backend process environment. Do not
put either secret or a `.env` file in Git. A Quick Tunnel URL changes after a
restart and is for testing only. Keep JupyterLab on port 38888 outside this
tunnel. Audio frames are handled transiently in process memory and are not
written to files, databases, or logs; transcripts are not generated.

An externally placed Twilio call still requires a configured Twilio number or
TwiML application pointing its voice webhook at
`https://<current-tunnel-host>/v1/twilio/voice`. The WebSocket URL is generated
from `PUBLIC_BASE_URL`. Verify the actual signed WebSocket handshake with a
controlled call before relying on this demo. Do not claim ML inference from the
mock result.

---

## Prioritized Backend Checklist

- [x] Initial monorepo skeleton (`backend/`, `contracts/`, `behavior-ml/`, `android/`, `docs/`) and `.gitignore`.
- [x] Version 1 contract models (`BehaviorWindow`, `BehaviorResult`, `AppEvent`, `VoiceTokenResponse`).
- [x] Document draft evidence-label vocabulary and risk meanings for team review.
- [ ] Implement FastAPI core application with `GET /health` and deterministic `WS /v1/app/call-events` mock.
- [ ] Configure secure public HTTPS/WSS ingress on LinuxONE for Twilio without exposing JupyterLab.
- [ ] Implement `POST /v1/voice/token` for Android Twilio Voice SDK registration.
- [x] Implement signed `POST /v1/twilio/voice` returning mocked `<Start><Stream track="inbound_track">` TwiML.
- [x] Implement signed `WS /v1/twilio/media` message handling with transient base64 payload decoding and no persistent audio buffer.
- [ ] Wire `behavior-ml`'s `analyze_window(request)` function into the audio pipeline.
- [ ] Validate privacy safeguards (memory-only audio, zero persistence, redacted logs) and capture LinuxONE execution proof.
