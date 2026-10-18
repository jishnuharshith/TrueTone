"""Local signed webhook check; no call, audio, Lookup charge, or secrets printed."""
from urllib.parse import urlencode
import sys
import xml.etree.ElementTree as ET

import httpx
from twilio.request_validator import RequestValidator

from backend.config import settings


def main() -> int:
    if not settings.twilio_auth_token or not settings.public_base_url:
        print("Missing TWILIO_AUTH_TOKEN or PUBLIC_BASE_URL in .env")
        return 2
    if settings.line_type_lookup:
        print("Set LINE_TYPE_LOOKUP=false before this smoke test to avoid a paid Lookup request.")
        return 2
    url = settings.public_base_url.rstrip("/") + "/v1/twilio/voice"
    params = {"CallSid": "CA-test-only", "From": "+15550000000"}
    signature = RequestValidator(settings.twilio_auth_token).compute_signature(url, params)
    try:
        response = httpx.post(
            "http://127.0.0.1:8000/v1/twilio/voice",
            content=urlencode(params),
            headers={"content-type": "application/x-www-form-urlencoded",
                     "x-twilio-signature": signature},
            timeout=5.0,
        )
    except httpx.HTTPError:
        print("Could not reach the local backend on 127.0.0.1:8000")
        return 1
    if response.status_code != 200:
        print(f"Webhook failed: HTTP {response.status_code}")
        return 1
    try:
        root = ET.fromstring(response.text)
    except ET.ParseError:
        print("Webhook did not return valid TwiML")
        return 1
    stream = root.find("./Start/Stream")
    if stream is None or not stream.attrib.get("url", "").startswith("wss://"):
        print("Webhook returned no secure Media Stream")
        return 1
    print("Signed webhook: OK")
    print("TwiML Media Stream: OK")
    print("Caller line type: unknown (Lookup disabled for this test)")
    print("Real incoming call and audio: not tested")
    return 0


if __name__ == "__main__":
    sys.exit(main())
