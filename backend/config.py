"""backend/config.py

Environment configuration for the FastAPI service.
Reads configuration from environment variables with safe defaults.
Zero hardcoded credentials.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


class Settings(BaseModel):
    service_name: str = "ibm-z-datathon-backend"
    version: str = "1.0.0"
    host: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    port: int = int(os.getenv("BACKEND_PORT", "8000"))
    environment: str = os.getenv("ENVIRONMENT", "development")
    mock_mode: bool = os.getenv("MOCK_MODE", "true").lower() in ("true", "1", "yes")
    live_audio_demo: bool = os.getenv("LIVE_AUDIO_DEMO", "false").lower() in ("true", "1", "yes")
    line_type_lookup: bool = os.getenv("LINE_TYPE_LOOKUP", "false").lower() in ("true", "1", "yes")

    # Twilio & Push Configuration (Names only; no defaults for secrets)
    twilio_account_sid: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    twilio_auth_token: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    # Trusted canonical HTTPS origin; set to the current Quick Tunnel hostname.
    public_base_url: str = os.getenv("PUBLIC_BASE_URL", "")
    twilio_api_key: str = os.getenv("TWILIO_API_KEY", "")
    twilio_api_secret: str = os.getenv("TWILIO_API_SECRET", "")
    twilio_twiml_app_sid: str = os.getenv("TWILIO_TWIML_APP_SID", "")
    twilio_push_credential_sid: str = os.getenv("TWILIO_PUSH_CREDENTIAL_SID", "")
    # Verified destination for temporary phone forwarding. Leave empty for mock response.
    twilio_forward_to: str = os.getenv("TWILIO_FORWARD_TO", "")

    # Android App WebSocket Authentication Token
    # CRITICAL: This is strictly a TEMPORARY mock/development credential.
    # - Do NOT place it in Android source code, an APK, Git, screenshots, or logs.
    # - The mock WebSocket endpoint remains private until the team agrees on how
    #   Android receives a short-lived, dynamically issued app-event credential.
    # - This static shared token is NOT production authentication.
    # - When unset or empty, WS /v1/app/call-events fails closed.
    app_auth_token: str = os.getenv("APP_AUTH_TOKEN", "")


settings = Settings()
