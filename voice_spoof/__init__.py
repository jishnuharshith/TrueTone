"""voice_spoof - Srikar's AI vs Human Voice Anti-Spoofing module.

Wraps the SE-ResNet model trained on WaveFake/ASVspoof datasets for
real-time detection of synthetic/cloned speech in caller audio streams.
"""

from .detector import VoiceSpoofResult, detect_pcm_window, get_detector

__all__ = ["VoiceSpoofResult", "detect_pcm_window", "get_detector"]
