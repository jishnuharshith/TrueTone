"""
Inference API and CLI for Voice Anti-Spoofing (Human vs AI Voice)
Designed for Real-Time Vishing Defense Systems.
IBM Z Datathon 2026
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Union, List

from config import AudioConfig, LABEL_BONAFIDE, LABEL_SPOOF, LABEL_NAMES
from dataset import load_audio_file, pad_or_trim
from features import extract_features
from models import get_resnet_antispoof, SpectroTemporalCNNLSTM


class VoiceSpoofDetector:
    """
    Real-time inference engine for detecting AI-synthesized & cloned speech.
    """
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_model.pt",
        device: str = None,
        threshold: float = None
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model_name = checkpoint.get("model_name", "resnet")
        self.feature_type = checkpoint.get("feature_type", "lfcc")
        self.audio_cfg = AudioConfig(feature_type=self.feature_type)

        # Set decision threshold
        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = checkpoint.get("optimal_threshold", 0.5)

        # Instantiate architecture
        if self.model_name.lower() in ("resnet", "se-resnet", "resnet-se"):
            self.model = get_resnet_antispoof(in_channels=3, num_classes=2)
        elif self.model_name.lower() in ("cnnlstm", "cnn_lstm"):
            self.model = SpectroTemporalCNNLSTM(in_channels=3, num_classes=2)
        else:
            raise ValueError(f"Unknown architecture: {self.model_name}")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict_audio_array(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        """
        Run inference on a raw 1D NumPy float32 waveform.
        """
        start_time = time.perf_counter()

        # Resample if needed
        if sr != self.audio_cfg.sample_rate:
            import scipy.signal
            num_resampled = int(len(audio) * self.audio_cfg.sample_rate / sr)
            audio = scipy.signal.resample(audio, num_resampled)

        # Pad or trim to configured duration
        audio_window = pad_or_trim(audio, self.audio_cfg.n_samples, random_crop=False)

        # Extract features (e.g. LFCC + deltas)
        feature = extract_features(
            audio_window,
            sr=self.audio_cfg.sample_rate,
            feature_type=self.audio_cfg.feature_type,
            with_deltas=self.audio_cfg.with_deltas
        )

        tensor_in = torch.from_numpy(feature).unsqueeze(0).float().to(self.device)  # (1, 3, F, T)

        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        human_prob = float(probs[LABEL_BONAFIDE])
        ai_prob = float(probs[LABEL_SPOOF])
        is_ai = bool(ai_prob >= self.threshold)

        # Determine Risk Level for Vishing Warning
        if ai_prob >= 0.85:
            risk_level = "CRITICAL"
            warning = "CRITICAL VISHING ALERT: Highly confident AI synthetic/cloned voice detected! Do NOT share OTP, passwords, or personal details."
        elif ai_prob >= self.threshold:
            risk_level = "HIGH"
            warning = "WARNING: Probable synthetic voice detected. Exercise extreme caution."
        elif ai_prob >= 0.35:
            risk_level = "SUSPICIOUS"
            warning = "ADVISORY: Acoustic anomalies detected. Confirm caller identity via an official callback channel."
        else:
            risk_level = "SAFE"
            warning = "Normal human vocal tract acoustic signatures verified."

        return {
            "prediction": LABEL_NAMES[LABEL_SPOOF if is_ai else LABEL_BONAFIDE],
            "is_ai": is_ai,
            "ai_confidence_pct": round(ai_prob * 100, 2),
            "human_confidence_pct": round(human_prob * 100, 2),
            "risk_level": risk_level,
            "vishing_warning": warning,
            "latency_ms": round(latency_ms, 2),
            "threshold_used": self.threshold
        }

    def predict_file(self, filepath: str) -> Dict[str, Any]:
        """
        Load audio from file and run anti-spoofing inference.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        audio = load_audio_file(filepath, target_sr=self.audio_cfg.sample_rate)
        result = self.predict_audio_array(audio, sr=self.audio_cfg.sample_rate)
        result["file"] = os.path.basename(filepath)
        return result


def main():
    parser = argparse.ArgumentParser(description="Voice Anti-Spoofing Inference (Human vs AI Voice)")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (.wav, .mp3, .flac)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--threshold", type=float, default=None, help="Custom decision threshold (0.0 to 1.0)")

    args = parser.parse_args()

    detector = VoiceSpoofDetector(checkpoint_path=args.checkpoint, threshold=args.threshold)
    res = detector.predict_file(args.audio)

    print("\n" + "=" * 55)
    print("      REAL-TIME VOICE ANTI-SPOOFING INFERENCE")
    print("=" * 55)
    print(f"File Tested:            {res['file']}")
    print(f"Result:                 {res['prediction']}")
    print(f"AI Spoof Confidence:    {res['ai_confidence_pct']}%")
    print(f"Human Confidence:       {res['human_confidence_pct']}%")
    print(f"Risk Assessment:        {res['risk_level']}")
    print(f"Inference Latency:      {res['latency_ms']} ms (Real-Time Ready)")
    print("-" * 55)
    print(f"System Advisory:        {res['vishing_warning']}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
