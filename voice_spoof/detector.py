"""voice_spoof/detector.py

Integration bridge between Srikar's AI vs Human Voice Anti-Spoofing model
and the TrueTone backend. Converts raw PCM (16-bit s16le, 16 kHz) into
a VoiceSpoofResult that complements the behavioral text-analysis pipeline.

Architecture: SE-ResNet trained on WaveFake / ASVspoof datasets.
Checkpoint: voice_spoof/checkpoints/best_model.pt (~34 MB, CPU-safe).
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("voice_spoof")

# Checkpoint path (relative to this file so it works regardless of CWD)
_DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "best_model.pt"


@dataclass(frozen=True, slots=True)
class VoiceSpoofResult:
    """Result from Srikar's AI-voice anti-spoofing model."""

    is_ai_voice: bool
    ai_confidence_pct: float    # 0-100
    human_confidence_pct: float  # 0-100
    risk_level: str             # SAFE | SUSPICIOUS | HIGH | CRITICAL
    vishing_warning: str
    latency_ms: float
    model_version: str = "voice-spoof-resnet-v1"


@lru_cache(maxsize=1)
def _load_detector(checkpoint_path: str):
    """Load Srikar's VoiceSpoofDetector; cached so model weights load once."""
    from .predict import VoiceSpoofDetector  # noqa: local import keeps torch optional
    return VoiceSpoofDetector(checkpoint_path=checkpoint_path)


def get_detector(checkpoint_path: Optional[str] = None):
    """Return the cached VoiceSpoofDetector, loading it on first call."""
    path = str(checkpoint_path or _DEFAULT_CHECKPOINT)
    return _load_detector(path)


def _pcm_s16le_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw s16le bytes to float32 in [-1, 1] range."""
    n_samples = len(pcm_bytes) // 2
    samples = struct.unpack(f"<{n_samples}h", pcm_bytes[:n_samples * 2])
    arr = np.array(samples, dtype=np.float32) / 32768.0
    return arr


def detect_pcm_window(
    pcm_s16le: bytes,
    sample_rate: int = 16000,
    checkpoint_path: Optional[str] = None,
) -> VoiceSpoofResult:
    """Run AI voice spoofing detection on a raw PCM window.

    Designed to be called from asyncio via asyncio.to_thread() since it
    does blocking PyTorch inference.

    Args:
        pcm_s16le: Raw 16-bit little-endian PCM bytes at 16 kHz.
        sample_rate: Input sample rate (default 16000; resampled if different).
        checkpoint_path: Override for the .pt checkpoint location.

    Returns:
        VoiceSpoofResult with AI confidence, risk level, and advisory text.
    """
    if not pcm_s16le or not any(pcm_s16le):
        # Silence / empty audio — return SAFE with zero confidence
        return VoiceSpoofResult(
            is_ai_voice=False,
            ai_confidence_pct=0.0,
            human_confidence_pct=100.0,
            risk_level="SAFE",
            vishing_warning="No caller audio detected.",
            latency_ms=0.0,
        )

    try:
        detector = get_detector(checkpoint_path)
        audio_f32 = _pcm_s16le_to_float32(pcm_s16le)
        raw = detector.predict_audio_array(audio_f32, sr=sample_rate)
        return VoiceSpoofResult(
            is_ai_voice=raw["is_ai"],
            ai_confidence_pct=raw["ai_confidence_pct"],
            human_confidence_pct=raw["human_confidence_pct"],
            risk_level=raw["risk_level"],
            vishing_warning=raw["vishing_warning"],
            latency_ms=raw["latency_ms"],
        )
    except Exception as exc:
        logger.warning("Voice spoof detection failed: %s", exc)
        return VoiceSpoofResult(
            is_ai_voice=False,
            ai_confidence_pct=0.0,
            human_confidence_pct=0.0,
            risk_level="SAFE",
            vishing_warning="Voice analysis temporarily unavailable.",
            latency_ms=0.0,
        )
