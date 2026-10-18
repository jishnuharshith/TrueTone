"""Experimental in-memory PCM -> whisper.cpp -> behavior-label path.

The CLI reads a WAV stream from stdin and writes only its transcript to stdout.
Neither the audio nor the transcript is written to a file by this module.
"""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import subprocess
from time import perf_counter_ns
from typing import Protocol
import wave

from .models import BehaviorResult, BehaviorWindow, LanguageHint
from .text import classify_text


class TranscriptionError(Exception):
    """An ASR failure without audio, transcript, or subprocess logs attached."""


class Transcriber(Protocol):
    @property
    def version(self) -> str: ...

    def transcribe_pcm(self, pcm_s16le: bytes, language_hint: LanguageHint) -> str: ...


@dataclass(frozen=True, slots=True)
class WhisperCliTranscriber:
    cli_path: str | Path
    model_path: str | Path
    threads: int = 2
    audio_context: int = 300
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.threads < 1 or self.audio_context < 0 or self.timeout_seconds <= 0:
            raise ValueError("threads and timeout must be positive; audio_context must be nonnegative")

    @property
    def version(self) -> str:
        return f"whisper-cli:{Path(self.model_path).stem}"

    def transcribe_pcm(self, pcm_s16le: bytes, language_hint: LanguageHint) -> str:
        """Transcribe 16 kHz mono s16le bytes using only subprocess pipes."""
        if not isinstance(pcm_s16le, bytes) or len(pcm_s16le) % 2:
            raise ValueError("pcm_s16le must be even-length bytes")
        if language_hint not in ("en", "hi", "auto"):
            raise ValueError("language_hint must be en, hi, or auto")
        if not pcm_s16le or not any(pcm_s16le):
            return ""

        wav_buffer = BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(pcm_s16le)

        command = [
            str(self.cli_path), "-m", str(self.model_path), "-f", "-",
            "-t", str(self.threads), "-l", language_hint,
            "-bs", "1", "-bo", "1", "-ac", str(self.audio_context),
            "-otxt", "-np",
        ]
        try:
            completed = subprocess.run(
                command,
                input=wav_buffer.getvalue(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise TranscriptionError("speech recognition unavailable") from None

        if completed.returncode != 0 or len(completed.stdout) > 16384:
            raise TranscriptionError("speech recognition failed")
        try:
            return " ".join(completed.stdout.decode("utf-8").split())
        except UnicodeDecodeError:
            raise TranscriptionError("speech recognition returned invalid text") from None


def analyze_window_with_asr(request: BehaviorWindow, transcriber: Transcriber) -> BehaviorResult:
    """Analyze one caller window; return no transcript, audio, or quoted content.

    Because the ASR has not been validated, a transcript with no rule match is
    insufficient evidence rather than a reassuring low-risk result.
    """
    start_ns = perf_counter_ns()
    if not isinstance(request, BehaviorWindow):
        raise TypeError("request must be a BehaviorWindow")

    assessment = None
    if request.pcm_s16le and any(request.pcm_s16le):
        try:
            transcript = transcriber.transcribe_pcm(request.pcm_s16le, request.language_hint)
        except TranscriptionError:
            transcript = ""
        if transcript:
            assessment = classify_text(transcript, request.language_hint)

    if assessment is None or assessment.risk_level in ("low", "insufficient_evidence"):
        language = assessment.language if assessment is not None else "unknown"
        confidence = 0.0
        risk_level = "insufficient_evidence"
        evidence_labels = []
    else:
        language = assessment.language
        confidence = assessment.confidence
        risk_level = assessment.risk_level
        evidence_labels = assessment.evidence_labels

    return BehaviorResult(
        schema_version="1",
        session_id=request.session_id,
        sequence=request.sequence,
        language=language,
        confidence=confidence,
        risk_level=risk_level,
        evidence_labels=evidence_labels,
        model_version=f"{transcriber.version}+text-prototype-v1",
        inference_ms=max(0, (perf_counter_ns() - start_ns) // 1_000_000),
    )
