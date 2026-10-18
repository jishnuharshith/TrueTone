"""
Configuration settings for Voice Anti-Spoofing (Human vs AI Speech Detection)
IBM Z Datathon 2026
"""

import os
from dataclasses import dataclass
import torch

@dataclass
class AudioConfig:
    sample_rate: int = 16000       # 16 kHz standard
    duration: float = 4.0          # Clip length in seconds (WaveFake 4s standard)
    n_samples: int = 64000         # sample_rate * duration
    
    # Feature extraction settings
    feature_type: str = "lfcc"     # "lfcc" (ASVspoof standard) | "mel" (Log-Mel) | "cqcc"
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    n_lfcc: int = 60
    n_filterbands: int = 70
    with_deltas: bool = True       # Include delta and delta-delta coefficients


@dataclass
class AugmentationConfig:
    # Telephony and VoIP simulation parameters for vishing defense
    enable: bool = True
    telephony_filter_prob: float = 0.1  # Moderate filter to retain vocoder phase cues
    noise_prob: float = 0.2             # Additive ambient background noise
    noise_snr_db_low: float = 15.0
    noise_snr_db_high: float = 35.0
    pitch_shift_prob: float = 0.1
    speed_perturb_prob: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 32
    num_epochs: int = 25
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 7
    loss_function: str = "cross_entropy"  # "cross_entropy" or "focal_loss"
    focal_gamma: float = 2.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0                  # 0 for Windows compatibility
    save_dir: str = os.path.join(os.path.dirname(__file__), "checkpoints")


# Standard labels
LABEL_BONAFIDE = 0  # Real / Human
LABEL_SPOOF = 1     # Fake / AI Synthesized / Cloned

LABEL_NAMES = {
    LABEL_BONAFIDE: "Human (Bonafide)",
    LABEL_SPOOF: "AI / Synthetic (Spoof)"
}
