"""
Real-Time Vishing Stream Monitor for Voice Anti-Spoofing
Simulates live phone call monitoring (or connects to live microphone)
and issues instant audio spoof alerts.
IBM Z Datathon 2026
"""

import os
import sys
import time
import argparse
import numpy as np

from config import AudioConfig, LABEL_BONAFIDE, LABEL_SPOOF
from dataset import load_audio_file
from predict import VoiceSpoofDetector


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class RealtimeVishingMonitor:
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_model.pt",
        window_sec: float = 3.0,
        hop_sec: float = 0.5,
        sr: int = 16000
    ):
        self.sr = sr
        self.window_size = int(window_sec * sr)
        self.hop_size = int(hop_sec * sr)
        self.detector = VoiceSpoofDetector(checkpoint_path=checkpoint_path)
        self.buffer = np.zeros(self.window_size, dtype=np.float32)

    def process_chunk(self, chunk: np.ndarray):
        """
        Ingests a chunk of audio, slides the buffer, and runs inference.
        """
        # Slide buffer
        chunk_len = len(chunk)
        if chunk_len >= self.window_size:
            self.buffer = chunk[-self.window_size:].astype(np.float32)
        else:
            self.buffer = np.roll(self.buffer, -chunk_len)
            self.buffer[-chunk_len:] = chunk

        # Run model inference on the active window
        res = self.detector.predict_audio_array(self.buffer, sr=self.sr)
        return res

    def monitor_stream_simulation(self, audio_file: str):
        """
        Simulates an incoming VoIP phone call by streaming an audio file chunk-by-chunk.
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

            if res["is_ai"]:
                alert_badge = "[!! VISHING ALERT !!]"
            elif risk == "SUSPICIOUS":
                alert_badge = "[? SUSPICIOUS ?]"
            else:
                alert_badge = "[OK HUMAN BONAFIDE]"

            bar_length = 20
            filled = int(bar_length * (ai_conf / 100.0))
            bar = "#" * filled + "-" * (bar_length - filled)

            print(
                f"T={sim_time:4.1f}s | {alert_badge:20s} | "
                f"AI Risk: [{bar}] {ai_conf:5.1f}% | "
                f"Inference: {latency:4.1f}ms"
            )
            time.sleep(0.08)  # Fast playback simulation

        print("\nCall ended. Monitoring terminated.")


def main():
    parser = argparse.ArgumentParser(description="Real-time Vishing Monitor")
    parser.add_argument("--stream_file", type=str, required=True, help="Path to audio file to stream")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Model checkpoint")

    args = parser.parse_args()
    monitor = RealtimeVishingMonitor(checkpoint_path=args.checkpoint)
    monitor.monitor_stream_simulation(args.stream_file)


if __name__ == "__main__":
    main()
