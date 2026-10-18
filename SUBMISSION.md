# TrueTone integrated hackathon prototype

## Verification (2026-10-18)

- Android `:app:assembleDebug`: BUILD SUCCESSFUL.
- Backend suite: 28 tests passed, including Android API contract checks.
- OTP-request smoke check: high, with OTP request / urgency / threat evidence.
- Protective-advice smoke check: low, with no evidence labels.
- APK has not yet been exercised on a physical device; configure Settings and run the demo steps below.

## Contents

- `android/`: Harshini's Compose frontend, connected to this backend. Home, analysis, session history and connection settings.
- `backend/`: FastAPI API and browser demo, Twilio routes and optional LinuxONE Whisper bridge.
- `behavior_ml/`: text risk prototype and speech bridge used by the demo.
- `contracts/`, `tests/`: event contracts and backend checks.

## Run the backend (PowerShell)

From the `TrueTone-backend` directory, using the existing environment:

```powershell
rtk proxy .venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, keep the HTTPS tunnel running:

```powershell
rtk proxy .\tools\cloudflared.exe tunnel --url http://127.0.0.1:8000
```

`cloudflared.exe` is a local tool, not included in source. Install Cloudflare Tunnel if setting up another computer. The printed HTTPS address changes after restarting a quick tunnel.

New backend environment setup: create `.venv` with Python, then install `backend/requirements.txt`. Scripted text requires `MOCK_MODE=true` (default) and no Twilio credentials. Keep `.env`, SSH keys and tokens private.

## Android

Open `android/` in Android Studio. Use the project Gradle wrapper, SDK 37 and Java 25 as specified by the original frontend project. Build/run on a device, or from `android/`:

```powershell
rtk proxy .\gradlew.bat :app:assembleDebug
```

Debug APK: `android/app/build/outputs/apk/debug/app-debug.apk`.

1. Open **Settings**, paste the current HTTPS tunnel origin without `/demo`, and tap **Save and test connection**.
2. Open **Detect**, submit an OTP-request sentence, and observe the backend risk, model and evidence.
3. Tap **Try protective advice**, then **Analyze text** to compare the response.
4. Save a result and open **History**. History is held only for this app session.
5. **Open microphone demo** opens the same backend's `/demo` page in the system browser. Grant microphone access there. **Fetch latest audio / call result** reads `/v1/demo/live-result` when returning to Android.

The browser demo at `<HTTPS origin>/demo` is also a submission fallback requiring no APK installation.

## API integration

| UI action | API |
| --- | --- |
| Save and test connection | GET `/health` |
| Analyze text | POST `/v1/demo/analyze-scripted-text`, JSON `text` and `language: auto` |
| Fetch audio/call result | GET `/v1/demo/live-result` |
| Open microphone | Browser `/demo`, microphone WebSocket managed by that page |

Android contains no Twilio credentials. HTTP failures and unavailable results are shown as status/errors, not successful low-risk detections. The old alternating preset risk results and preset 92/96 percent scores were removed.

## Scope and remaining work

- Scripted analysis uses the text rule prototype. Its score is not a calibrated probability.
- Browser audio uses the configured LinuxONE Whisper bridge plus text rules; it requires the existing SSH/Whisper setup and `LIVE_AUDIO_DEMO=true`.
- Android opens browser microphone capture; it does not capture cellular call audio or automatically intercept incoming calls.
- The trained XGBoost classifier and AI-generated voice detector are not connected to this submitted UI.
- Twilio routes exist, but a successful end-to-end phone speech test has not been established. Do not present call completion as successful analysis.
- Optional line-type Lookup describes a number's line type, not whether a caller is a scammer or using an AI voice.
- Live result state is shared by the demo server. This is a controlled single-user demo, not a multi-user production service.
- The frontend source was imported from `origin/harshini/frontend-upload` into `android/`; backend work remains on `shaun/backend-integration`.

Do not include `.env`, `.venv`, keys, Gradle caches or runtime logs in a submission. Submit source and, after a successful build, the debug APK separately.
