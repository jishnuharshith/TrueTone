"""Optional Twilio Lookup signal for the inbound caller's phone number."""
import re
from urllib.parse import quote

import httpx

ALLOWED_TYPES = frozenset({
    "mobile", "landline", "fixedVoip", "nonFixedVoip", "personal",
    "tollFree", "premium", "sharedCost", "uan", "voicemail", "pager", "unknown",
})


async def lookup_line_type(number: str, username: str, password: str) -> str:
    """Return a line type or unknown. Never log or retain the caller number."""
    if not re.fullmatch(r"\+[1-9]\d{6,14}", number or "") or not username or not password:
        return "unknown"
    url = "https://lookups.twilio.com/v2/PhoneNumbers/" + quote(number, safe="")
    try:
        async with httpx.AsyncClient(timeout=3.0, auth=(username, password)) as client:
            response = await client.get(url, params={"Fields": "line_type_intelligence"})
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError):
        return "unknown"
    line = payload.get("line_type_intelligence")
    value = line.get("type") if isinstance(line, dict) else None
    return value if value in ALLOWED_TYPES else "unknown"
