"""Transient Twilio mu-law caller audio to LinuxONE Whisper and behavior labels."""
import audioop
import base64
import os
import subprocess
import wave
from io import BytesIO
from pathlib import Path
from dataclasses import dataclass, field
from uuid import UUID

from behavior_ml.asr import analyze_window_with_asr, TranscriptionError
from behavior_ml.models import BehaviorWindow

WINDOW_BYTES = 32000  # Four seconds of 8 kHz, 8-bit mu-law.


@dataclass
class CallerAudioWindow:
    data: bytearray = field(default_factory=bytearray)
    sequence: int = 0

    def add_payload(self, payload: str) -> bytes | None:
        chunk = base64.b64decode(payload, validate=True)
        if len(chunk) > 4096:
            raise ValueError("Oversized media frame")
        self.data.extend(chunk)
        if len(self.data) < WINDOW_BYTES:
            return None
        window = bytes(self.data[:WINDOW_BYTES])
        del self.data[:WINDOW_BYTES]
        return window


def mulaw_to_pcm16(window: bytes) -> bytes:
    pcm8 = audioop.ulaw2lin(window, 2)
    pcm16, _ = audioop.ratecv(pcm8, 2, 1, 8000, 16000, None)
    return pcm16


class LinuxOneWhisper:
    version = "linuxone-whisper-small"

    def transcribe_pcm(self, pcm_s16le: bytes, language_hint: str) -> str:
        key = os.getenv("LINUXONE_SSH_KEY", "")
        if not key or not Path(key).is_file():
            raise TranscriptionError("LinuxONE key unavailable")
        audio = BytesIO()
        with wave.open(audio, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(pcm_s16le)
        command = ["ssh", "-i", key, "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                   "linux1@148.100.109.138",
                   "/home/linux1/shaun-whisper-cpp/build-z16/bin/whisper-cli",
                   "-m", "/home/linux1/shaun-whisper-cpp/models/ggml-small.bin",
                   "-f", "-", "-t", "2", "-l", language_hint,
                   "-bs", "1", "-bo", "1", "-ac", "300", "-otxt", "-np"]
        try:
            result = subprocess.run(command, input=audio.getvalue(), capture_output=True,
                                    timeout=60, check=False)
        except (OSError, subprocess.TimeoutExpired):
            raise TranscriptionError("LinuxONE transcription unavailable") from None
        if result.returncode or len(result.stdout) > 16384:
            raise TranscriptionError("LinuxONE transcription failed")
        transcript = " ".join(result.stdout.decode("utf-8", errors="replace").split())
        # Whisper sometimes writes spoken Hindi in Urdu script. Normalize the
        # observed OTP/request spellings before the existing behavior rules.
        return transcript.replace("اوٹیپی", "otp").replace("بداو", "batao")


def assess_caller_window(session_id: str, sequence: int, mulaw: bytes):
    UUID(session_id)
    language = os.getenv("LIVE_AUDIO_LANGUAGE", "hi")
    if language not in {"en", "hi", "auto"}:
        language = "auto"
    request = BehaviorWindow("1", session_id, sequence, language, 16000,
                             mulaw_to_pcm16(mulaw), sequence * 4000, (sequence + 1) * 4000)
    return analyze_window_with_asr(request, LinuxOneWhisper())
