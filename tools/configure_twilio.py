"""Save the Twilio secret privately, without shell history or terminal echo."""
from getpass import getpass
from pathlib import Path
import os

root = Path(__file__).resolve().parents[1]
token = getpass("Paste current Twilio Auth Token (hidden): ").strip()
if not token or not token.isalnum():
    raise SystemExit("Expected a nonempty alphanumeric Auth Token.")
path = root / ".env"
existing = path.read_text() if path.exists() else "MOCK_MODE=true\n"
lines = [line for line in existing.splitlines() if not line.startswith("TWILIO_AUTH_TOKEN=")]
path.write_text("\n".join(lines) + "\nTWILIO_AUTH_TOKEN=" + token + "\n")
os.chmod(path, 0o600)
print("Saved locally to ignored .env. Restart the backend to load it.")
