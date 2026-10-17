"""
Automated Test Cases for Voice Anti-Spoofing & Vishing Detection
IBM Z Datathon 2026
"""

import os
import sys
import time
import unittest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict import VoiceSpoofDetector
from config import AudioConfig, LABEL_BONAFIDE, LABEL_SPOOF, LABEL_NAMES
from features import extract_features, extract_tabular_acoustic_features
from dataset import load_audio_file


class VoiceAntiSpoofingTestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pt")
        if not os.path.exists(cls.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {cls.checkpoint_path}")
        
        print(f"\n[Setup] Initializing VoiceSpoofDetector with: {cls.checkpoint_path}")
        cls.detector = VoiceSpoofDetector(checkpoint_path=cls.checkpoint_path)
        
        # Audio test sample paths
        cls.real_sample = r"E:\IBM\dataset\_sources_wavefake\real_voice\LJ001-0004.wav"
        cls.fake_sample = r"E:\IBM\dataset\_sources_wavefake\fake_voice\LJ001-0004_generated.wav"

    def test_case_01_feature_extraction_dimensions(self):
        """Test Case 1: Feature Extraction Pipeline (LFCC + Deltas Shape)"""
        print("\n--- Test Case 1: Feature Extractor Shape & CMVN ---")
        dummy_audio = np.random.randn(64000).astype(np.float32)
        feature = extract_features(dummy_audio, sr=16000, feature_type="lfcc", with_deltas=True)
        
        # Expect (3 channels, 60 LFCC bins, Time frames)
        self.assertEqual(feature.ndim, 3)
        self.assertEqual(feature.shape[0], 3)
        self.assertEqual(feature.shape[1], 60)
        self.assertGreater(feature.shape[2], 200)
        print(f"  [PASS] Feature shape verified: {feature.shape} (Channels=3, Bins=60, Frames={feature.shape[2]})")

    def test_case_02_human_voice_verification(self):
        """Test Case 2: Authentic Human Speech Verification (Bonafide)"""
        print("\n--- Test Case 2: Human Voice Verification ---")
        if not os.path.exists(self.real_sample):
            self.skipTest(f"Audio file not found: {self.real_sample}")

        result = self.detector.predict_file(self.real_sample)
        print(f"  File: {os.path.basename(self.real_sample)}")
        print(f"  Result: {result['prediction']}")
        print(f"  Human Confidence: {result['human_confidence_pct']}% | AI Risk: {result['ai_confidence_pct']}%")
        print(f"  Risk Level: {result['risk_level']} | Latency: {result['latency_ms']} ms")
        
        self.assertFalse(result["is_ai"], "Human sample was wrongly flagged as AI!")
        self.assertEqual(result["risk_level"], "SAFE")
        self.assertGreater(result["human_confidence_pct"], 80.0)
        print("  [PASS] Verified Authentic Human Voice correctly classified as SAFE.")

    def test_case_03_ai_spoof_detection(self):
        """Test Case 3: AI Synthetic Voice Detection (Spoof)"""
        print("\n--- Test Case 3: AI Synthetic Voice Detection ---")
        if not os.path.exists(self.fake_sample):
            self.skipTest(f"Audio file not found: {self.fake_sample}")

        result = self.detector.predict_file(self.fake_sample)
        print(f"  File: {os.path.basename(self.fake_sample)}")
        print(f"  Result: {result['prediction']}")
        print(f"  AI Spoof Confidence: {result['ai_confidence_pct']}% | Human Confidence: {result['human_confidence_pct']}%")
        print(f"  Risk Level: {result['risk_level']} | Latency: {result['latency_ms']} ms")
        print(f"  Warning: {result['vishing_warning']}")
        
        self.assertTrue(result["is_ai"], "AI synthetic sample was not caught!")
        self.assertIn(result["risk_level"], ["HIGH", "CRITICAL"])
        self.assertGreater(result["ai_confidence_pct"], 80.0)
        print("  [PASS] AI voice clone caught with High/Critical risk assessment.")

    def test_case_04_streaming_raw_array_latency(self):
        """Test Case 4: Real-Time Latency Benchmark on Raw Audio Waveform"""
        print("\n--- Test Case 4: Real-Time Stream Latency Benchmark ---")
        dummy_audio = np.random.randn(64000).astype(np.float32)
        
        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            res = self.detector.predict_audio_array(dummy_audio, sr=16000)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        avg_latency = np.mean(latencies)
        print(f"  Average Inference Latency: {avg_latency:.2f} ms")
        self.assertLess(avg_latency, 250.0, f"Inference too slow ({avg_latency} ms) for real-time operation!")
        print("  [PASS] Latency verified within real-time streaming constraints.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
