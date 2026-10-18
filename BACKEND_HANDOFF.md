# Backend integration work

Starting point: `origin/aadvik/backend-upload`, commit `c02c91f`.
Working branch: `shaun/backend-integration`.

## Implemented foundation

- Health and contract fixture endpoints.
- Authenticated app WebSocket with deterministic mock events.
- Signed Twilio voice webhook and media WebSocket.
- Separate trial-compatible Twilio phone route: `/v1/twilio/trial-voice` uses
  Twilio `<Gather input="speech">`, then a signed callback classifies the
  recognized text. It does not run Whisper or analyze the call's raw audio.
- Transient Twilio inbound-audio windows can run Whisper plus the text rules when `LIVE_AUDIO_DEMO=true`.
- Optional Twilio Lookup v2 Line Type Intelligence runs on the signed voice webhook when `LINE_TYPE_LOOKUP=true`. It returns a caller-number line category, never an AI-voice verdict. It is disabled by default because Lookup may be billable.
- Mock app events are now disabled when `MOCK_MODE=false`.

## Remaining integration

1. Align frontend events and ML result schemas with the contracts.
2. Harden the proof-of-concept live audio path for simultaneous calls and longer latency; current demo state is process-global.
3. Validate the Twilio trial phone route with a permitted verified-number call
   and compare behavior results to a consented script. Trial accounts block
   `<Stream>`, so the Media Streams path requires a full-access account.
4. Route results to the authenticated client for the corresponding call; clean up on disconnect.
5. Implement Voice SDK token issuance and call routing after checking the frontend flow.
6. Run local integration tests, then a controlled Twilio call on LinuxONE.

Voice spoofing is a separate signal. The checkpoint reviewed on main missed 28 of 30 committed spoof samples, so it needs correction and evaluation before driving alerts.

## Local verification

From this checkout, install `backend/requirements.txt` in `.venv`, then run:

```powershell
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Mock event clients require `APP_AUTH_TOKEN` supplied through the environment. Live calling still needs Twilio configuration and public HTTPS/WSS ingress. Calls must reach the Twilio number directly or via carrier forwarding from the user's mobile number; Twilio cannot inspect a separate mobile call without that routing. Trial inbound callers must be verified. A VoIP line type is only caller-number metadata; it neither proves a scam nor detects synthesized speech.
