"""
Real-Time Live Call Anti-Spoofing & Vishing Defense Engine
TrueTone - IBM Z Datathon 2026

Supports:
1. Live Hardware Audio Streaming: Listens directly to live microphone / headset / call audio.
2. Device Selection: Choose any input device (Microphone, USB Headset, Virtual Cable, Stereo Mix).
3. Headless App Streaming API: Real-time WebSocket & REST service for mobile/desktop app integration.
4. Audio File Simulation: Test playback of recorded phone calls.
"""

import os
import sys
import time
import queue
import argparse
import numpy as np
import torch
import json
import base64
import scipy.signal
from typing import Dict, Any, Optional

from config import AudioConfig, LABEL_BONAFIDE, LABEL_SPOOF
from predict import VoiceSpoofDetector
from dataset import load_audio_file, pad_or_trim

# Ensure utf-8 stdout on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Optional audible alert on Windows
try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False


def build_mulaw_expansion_table() -> np.ndarray:
    """
    Constructs the standard ITU-T G.711 mu-law expansion table for telephony audio streams.
    Converts 8-bit telephony mu-law bytes directly to normalized float32.
    """
    table = np.zeros(256, dtype=np.float32)
    for i in range(256):
        val = ~i
        sign = (val & 0x80)
        exponent = (val >> 4) & 0x07
        mantissa = val & 0x0F
        sample = ((mantissa << 3) + 0x84) << exponent
        sample -= 0x84
        if not sign:
            sample = -sample
        table[i] = sample / 32768.0
    return table


class RealtimeVishingMonitor:
    """
    Continuous real-time audio stream monitor for live phone calls and voice streams.
    Slides a 3-second temporal window with 0.5-second hops, classifying incoming
    audio frames in under 40 milliseconds.
    """
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_model.pt",
        window_sec: float = 3.0,
        hop_sec: float = 0.5,
        sr: int = 16000,
        vad_threshold: float = 0.016
    ):
        self.sr = sr
        self.window_size = int(window_sec * sr)
        self.hop_size = int(hop_sec * sr)
        self.detector = VoiceSpoofDetector(checkpoint_path=checkpoint_path)
        self.buffer = np.zeros(self.window_size, dtype=np.float32)

        # Robust Voice Activity Detection (VAD) parameters
        self.noise_floor = 0.006
        self.min_speech_rms = vad_threshold
        self.min_speech_peak = 0.038
        self.speech_hangover = 0  # Hold active speech across micro-pauses between words

    @staticmethod
    def list_devices():
        """
        Lists all available hardware and virtual audio input devices on the system.
        """
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            print("\n" + "=" * 76)
            print("         AVAILABLE AUDIO INPUT DEVICES (FOR LIVE CALL LISTENING)")
            print("=" * 76)
            for idx, d in enumerate(devices):
                if d["max_input_channels"] > 0:
                    default_marker = " [DEFAULT - MIC/HEADSET]" if idx == sd.default.device[0] else ""
                    name = d["name"]
                    is_stereo_mix = "stereo mix" in name.lower() or "wave out" in name.lower()
                    mix_marker = " <-- [RECOMMENDED FOR ZOOM / TEAMS / CALL APPS]" if is_stereo_mix else ""
                    print(f"  [{idx:2d}] {name} (Channels: {d['max_input_channels']}){default_marker}{mix_marker}")
            print("=" * 76)
            print("💡 HOW TO LISTEN IN LIVE CALLS:")
            print("  1. Phone Call on Speakerphone or Desktop Mic:")
            print("     Run: py realtime_detector.py --live")
            print("  2. Call from Desktop Software (Zoom, MS Teams, WhatsApp, Discord):")
            print("     Choose the 'Stereo Mix' device number from above and run:")
            print("     py realtime_detector.py --live --device <ID>")
            print("  3. Headless Streaming API for App Integration (WebSocket & REST):")
            print("     Run: py realtime_detector.py --api --port 8000")
            print("     Stream live call chunks to ws://localhost:8000/ws/live-call")
            print("=" * 76 + "\n")
        except ImportError:
            print("[Error] sounddevice library not installed. Run: pip install sounddevice")

    def process_chunk(self, chunk: np.ndarray, vad_sensitivity: float = 1.0) -> Dict[str, Any]:
        """
        Ingests a chunk of live audio (1D float32 array), updates the rolling buffer,
        computes Voice Activity Detection (VAD) energy, and runs inference.
        """
        if chunk.ndim > 1:
            chunk = np.mean(chunk, axis=-1)

        chunk = chunk.astype(np.float32)

        # Remove DC bias offset from microphone / sound card
        chunk = chunk - float(np.mean(chunk))

        # Slide buffer
        chunk_len = len(chunk)
        if chunk_len >= self.window_size:
            self.buffer = chunk[-self.window_size:]
        else:
            self.buffer = np.roll(self.buffer, -chunk_len)
            self.buffer[-chunk_len:] = chunk

        # Voice Activity Detection (VAD) Energy Metrics
        chunk_rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-12))
        chunk_peak = float(np.max(np.abs(chunk)))
        buffer_rms = float(np.sqrt(np.mean(self.buffer ** 2) + 1e-12))
        buffer_peak = float(np.max(np.abs(self.buffer)))

        # Update ambient noise floor estimate slowly on quiet frames
        if chunk_rms < 0.025:
            self.noise_floor = 0.96 * self.noise_floor + 0.04 * chunk_rms

        # Dynamic speech threshold: speech must rise significantly above the room noise floor
        adaptive_threshold = max(self.min_speech_rms, self.noise_floor * 2.8) / max(0.2, vad_sensitivity)
        peak_threshold = self.min_speech_peak / max(0.2, vad_sensitivity)

        is_current_speech = (chunk_rms >= adaptive_threshold * 0.75) and (chunk_peak >= peak_threshold * 0.75)
        is_buffer_speech = (buffer_rms >= adaptive_threshold) and (buffer_peak >= peak_threshold)

        if is_current_speech or is_buffer_speech:
            self.speech_hangover = 3  # Hold active state across micro-pauses between words
        elif self.speech_hangover > 0:
            self.speech_hangover -= 1

        is_voice_active = (self.speech_hangover > 0) and is_buffer_speech

        if not is_voice_active:
            return {
                "prediction": "Silence / Ambient Noise",
                "is_ai": False,
                "ai_confidence_pct": 0.0,
                "human_confidence_pct": 100.0,
                "risk_level": "SAFE",
                "vishing_warning": "No active voice detected. Ambient soundscape monitored.",
                "latency_ms": 0.5,
                "rms_energy": float(buffer_rms),
                "peak_energy": float(buffer_peak),
                "is_silence": True
            }

        # Run model inference on active speech window
        res = self.detector.predict_audio_array(self.buffer, sr=self.sr)
        res["rms_energy"] = float(buffer_rms)
        res["peak_energy"] = float(buffer_peak)
        res["is_silence"] = False
        return res

    def monitor_live_audio(
        self,
        device_idx: Optional[int] = None,
        sound_alert: bool = True
    ):
        """
        Listens directly to the live microphone, headset, or call audio device in real time.
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise RuntimeError("sounddevice is required for live audio capture. Run: pip install sounddevice")

        chosen_device = device_idx if device_idx is not None else sd.default.device[0]
        device_info = sd.query_devices(chosen_device)
        dev_name = device_info["name"]

        print("\n" + "=" * 76)
        print("    🛡️  TRUETONE: REAL-TIME LIVE CALL ANTI-VISHING DEFENSE MONITOR")
        print("=" * 76)
        print(f"  Hardware Input Device:  [{chosen_device}] {dev_name}")
        print(f"  Sampling Frequency:     {self.sr} Hz")
        print(f"  Temporal Window:        {self.window_size / self.sr:.1f}s (hop: {self.hop_size / self.sr:.2f}s)")
        print(f"  Speech VAD Threshold:   RMS >= {self.min_speech_rms:.3f} | Peak >= {self.min_speech_peak:.3f}")
        print(f"  Audible Alarms:         {'Enabled (Sound Beep on Attack)' if sound_alert else 'Disabled'}")
        print("  Press Ctrl+C at any time to end live call monitoring.")
        print("-" * 76 + "\n")

        audio_q: queue.Queue = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                pass
            audio_q.put(indata.copy().squeeze(-1))

        start_time = time.time()
        last_alert_time = 0.0

        try:
            with sd.InputStream(
                device=device_idx,
                channels=1,
                samplerate=self.sr,
                blocksize=self.hop_size,
                dtype="float32",
                callback=audio_callback
            ):
                print("[*] LIVE CALL STREAM INITIALIZED. LISTENING FOR CALLER VOICE...\n")
                while True:
                    try:
                        chunk = audio_q.get(timeout=1.0)
                    except queue.Empty:
                        continue

                    elapsed = time.time() - start_time
                    mins, secs = int(elapsed // 60), int(elapsed % 60)

                    res = self.process_chunk(chunk)
                    ai_conf = res["ai_confidence_pct"]
                    risk = res["risk_level"]
                    latency = res["latency_ms"]
                    is_silence = res.get("is_silence", False)

                    # Badge formatting
                    if is_silence:
                        badge = "[ 🎙️ SILENCE / WAITING  ]"
                        bar = "░" * 20
                        print(
                            f"[{mins:02d}:{secs:02d}] | {badge} | "
                            f"Ambient Noise Floor (Waiting for speech...) | "
                            f"Latency: {latency:4.1f}ms",
                            flush=True
                        )
                        continue

                    if res["is_ai"]:
                        badge = "[🚨 CRITICAL VISHING! 🚨]"
                        bar = "█" * int(20 * (ai_conf / 100.0)) + "░" * (20 - int(20 * (ai_conf / 100.0)))
                    elif risk == "HIGH":
                        badge = "[⚠️ HIGH SUSPICION ⚠️]"
                        bar = "█" * int(20 * (ai_conf / 100.0)) + "░" * (20 - int(20 * (ai_conf / 100.0)))
                    elif risk == "SUSPICIOUS":
                        badge = "[? SUSPICIOUS AUDIO ?]"
                        bar = "▒" * int(20 * (ai_conf / 100.0)) + "░" * (20 - int(20 * (ai_conf / 100.0)))
                    else:
                        badge = "[✅ AUTHENTIC HUMAN  ]"
                        bar = "█" * int(20 * (ai_conf / 100.0)) + "░" * (20 - int(20 * (ai_conf / 100.0)))

                    print(
                        f"[{mins:02d}:{secs:02d}] | {badge} | "
                        f"AI Risk: [{bar}] {ai_conf:5.1f}% | "
                        f"Latency: {latency:4.1f}ms",
                        flush=True
                    )

                    # Trigger audible warning sound when critical vishing is detected
                    if res["is_ai"] and sound_alert and WINSOUND_AVAILABLE:
                        curr_t = time.time()
                        if curr_t - last_alert_time > 1.5:
                            winsound.Beep(1500, 250)
                            last_alert_time = curr_t

        except KeyboardInterrupt:
            print("\n\n[*] Live call monitoring terminated by user.")
            total_duration = time.time() - start_time
            print(f"[*] Total call session duration: {int(total_duration // 60):02d}:{int(total_duration % 60):02d}")

    def monitor_stream_simulation(self, audio_file: str):
        """
        Simulates an incoming VoIP phone call by streaming a pre-recorded audio file chunk-by-chunk.
        """
        print(f"\n=======================================================")
        print(f"  REAL-TIME VISHING CALL MONITOR (INCOMING CALL SIMULATION)")
        print(f"  Streaming: {os.path.basename(audio_file)}")
        print(f"=======================================================\n")

        full_audio = load_audio_file(audio_file, target_sr=self.sr)
        total_samples = len(full_audio)
        current_idx = 0

        chunk_samples = self.hop_size
        sim_time = 0.0

        while current_idx < total_samples:
            chunk = full_audio[current_idx : current_idx + chunk_samples]
            current_idx += chunk_samples
            sim_time += len(chunk) / self.sr

            res = self.process_chunk(chunk)
            ai_conf = res["ai_confidence_pct"]
            risk = res["risk_level"]
            latency = res["latency_ms"]

            if res.get("is_silence", False):
                alert_badge = "[ 🎙️ SILENCE / WAITING  ]"
                bar = "░" * 20
            elif res["is_ai"]:
                alert_badge = "[🚨 VISHING ALERT! 🚨]"
                bar = "█" * int(20 * (ai_conf / 100.0)) + "░" * (20 - int(20 * (ai_conf / 100.0)))
            elif risk == "SUSPICIOUS":
                alert_badge = "[? SUSPICIOUS AUDIO ?]"
                bar = "▒" * int(20 * (ai_conf / 100.0)) + "░" * (20 - int(20 * (ai_conf / 100.0)))
            else:
                alert_badge = "[✅ AUTHENTIC HUMAN  ]"
                bar = "█" * int(20 * (ai_conf / 100.0)) + "░" * (20 - int(20 * (ai_conf / 100.0)))

            print(
                f"T={sim_time:4.1f}s | {alert_badge} | "
                f"AI Risk: [{bar}] {ai_conf:5.1f}% | "
                f"Inference: {latency:4.1f}ms"
            )
            time.sleep(0.08)

        print("\nCall ended. Monitoring terminated.")


def start_server(checkpoint_path: str = "checkpoints/best_model.pt", host: str = "0.0.0.0", port: int = 8000, vad_threshold: float = 0.016):
    """
    Launches a headless FastAPI WebSocket & REST server for real-time live calls.
    Designed for seamless integration into mobile apps, desktop apps, and telephony systems.
    Clients stream raw audio chunks via WebSocket (/ws/live-call) or REST (/api/predict_chunk).
    """
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from pydantic import BaseModel

    app = FastAPI(
        title="TrueTone Live Vishing Defense API",
        version="2.0",
        description="Headless Live Audio Stream Vishing Detection Engine for App Integration"
    )
    monitor = RealtimeVishingMonitor(checkpoint_path=checkpoint_path, vad_threshold=vad_threshold)
    mulaw_table = build_mulaw_expansion_table()

    class AudioChunkPayload(BaseModel):
        audio_base64: str
        sample_rate: int = 16000

    @app.get("/health")
    def health():
        return {
            "status": "active",
            "service": "TrueTone Live Vishing Engine (App Backend)",
            "model_device": monitor.detector.device,
            "sample_rate": monitor.sr,
            "vad_threshold": monitor.min_speech_rms
        }

    @app.post("/api/predict_chunk")
    def predict_chunk(payload: AudioChunkPayload):
        """
        REST endpoint for mobile or desktop apps to post a single audio chunk (base64 16-bit PCM).
        """
        raw_bytes = base64.b64decode(payload.audio_base64)
        chunk = (np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)) / 32768.0
        if payload.sample_rate != 16000:
            chunk = scipy.signal.resample_poly(chunk, 16000, payload.sample_rate).astype(np.float32)
        return monitor.process_chunk(chunk)

    @app.websocket("/ws/live-call")
    async def websocket_call_endpoint(websocket: WebSocket):
        """
        WebSocket endpoint for real-time live call streaming.
        Accepts:
        - Raw binary Float32 / Int16 PCM chunks
        - JSON with base64 encoded audio chunks
        - Twilio Media Streams JSON ({"event": "media", ...})
        """
        await websocket.accept()
        try:
            while True:
                message = await websocket.receive()
                if "bytes" in message and message["bytes"]:
                    data = message["bytes"]
                    # Determine whether raw buffer is float32 or int16
                    if len(data) % 4 == 0:
                        chunk = np.frombuffer(data, dtype=np.float32)
                        # Sanity check: float32 audio should rarely exceed 10.0
                        if len(chunk) > 0 and np.max(np.abs(chunk)) > 10.0:
                            chunk = (np.frombuffer(data, dtype=np.int16).astype(np.float32)) / 32768.0
                    else:
                        chunk = (np.frombuffer(data, dtype=np.int16).astype(np.float32)) / 32768.0

                    if len(chunk) > 0:
                        res = monitor.process_chunk(chunk)
                        await websocket.send_json(res)

                elif "text" in message and message["text"]:
                    payload_json = json.loads(message["text"])
                    # Check for Twilio Media Streams format
                    event_type = payload_json.get("event")
                    if event_type == "media":
                        media = payload_json.get("media", {})
                        raw_b64 = media.get("payload", "")
                        raw_bytes = base64.b64decode(raw_b64)
                        # Twilio sends 8kHz G.711 mu-law: expand and resample to 16kHz
                        chunk_8k = mulaw_table[np.frombuffer(raw_bytes, dtype=np.uint8)]
                        chunk_16k = scipy.signal.resample_poly(chunk_8k, 2, 1).astype(np.float32)
                        res = monitor.process_chunk(chunk_16k)
                        await websocket.send_json(res)
                    elif "audio_base64" in payload_json:
                        raw_bytes = base64.b64decode(payload_json["audio_base64"])
                        sr = payload_json.get("sample_rate", 16000)
                        chunk = (np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)) / 32768.0
                        if sr != 16000:
                            chunk = scipy.signal.resample_poly(chunk, 16000, sr).astype(np.float32)
                        res = monitor.process_chunk(chunk)
                        await websocket.send_json(res)

        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    print("\n" + "=" * 76)
    print("  🛡️ TRUETONE LIVE CALL ENGINE (HEADLESS STREAMING API)")
    print("=" * 76)
    print(f"  WebSocket Endpoint: ws://localhost:{port}/ws/live-call")
    print(f"  REST Endpoint:      http://localhost:{port}/api/predict_chunk")
    print(f"  Health Check:       http://localhost:{port}/health")
    print("=" * 76 + "\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main():
    parser = argparse.ArgumentParser(description="TrueTone Real-Time Voice Anti-Spoofing & Live Call Vishing Defense")
    parser.add_argument("--live", action="store_true", help="Listen directly to live hardware microphone/call stream")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index (use --list-devices to view)")
    parser.add_argument("--list-devices", action="store_true", help="List all available audio input devices")
    parser.add_argument("--server", "--api", action="store_true", dest="server", help="Launch headless WebSocket & REST API service for app integration")
    parser.add_argument("--port", type=int, default=8000, help="API server port (default: 8000)")
    parser.add_argument("--vad_threshold", type=float, default=0.016, help="Energy RMS threshold for speech detection (default: 0.016)")
    parser.add_argument("--stream_file", type=str, default=None, help="Simulate a live call by streaming an audio file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Model checkpoint path")
    parser.add_argument("--no-sound", action="store_true", help="Disable audible beeps on attacks")

    args = parser.parse_args()

    if args.list_devices:
        RealtimeVishingMonitor.list_devices()
        return

    if args.server:
        start_server(checkpoint_path=args.checkpoint, port=args.port, vad_threshold=args.vad_threshold)
        return

    monitor = RealtimeVishingMonitor(checkpoint_path=args.checkpoint, vad_threshold=args.vad_threshold)

    if args.live:
        monitor.monitor_live_audio(device_idx=args.device, sound_alert=not args.no_sound)
    elif args.stream_file:
        monitor.monitor_stream_simulation(args.stream_file)
    else:
        # Default to live mode if no file provided
        print("[*] No audio file specified. Entering live microphone listening mode...")
        print("[*] (Run with --help to see all options or --server for app streaming API)\n")
        monitor.monitor_live_audio(device_idx=args.device, sound_alert=not args.no_sound)


if __name__ == "__main__":
    main()


