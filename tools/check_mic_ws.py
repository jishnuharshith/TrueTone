"""Check public browser microphone WebSocket without sending caller audio."""
import asyncio
import argparse
import websockets


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("origin", help="Current HTTPS tunnel origin, without /demo")
    origin = parser.parse_args().origin.rstrip("/")
    if not origin.startswith("https://"):
        parser.error("origin must start with https://")
    url = "wss://" + origin[len("https://"):] + "/v1/demo/microphone"
    try:
        async with websockets.connect(url, open_timeout=10) as ws:
            print("public_microphone_websocket: connected")
    except Exception as exc:
        print("public_microphone_websocket:", type(exc).__name__, str(exc)[:160])


asyncio.run(main())
