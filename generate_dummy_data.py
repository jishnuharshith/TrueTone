"""
Synthetic Audio Data Generator for Pipeline Verification
Generates realistic acoustic waveforms mimicking:
1. Bonafide (Human) Speech: Glottal formant synthesis with natural vocal jitter & vibrato
2. Spoofed (AI) Speech: Vocoder phase distortion, mechanical pitch rigidity & vocoder artifacts

Enables immediate end-to-end verification of training, validation, evaluation, and prediction!
"""

import os
import argparse
import numpy as np
import scipy
import scipy.signal
import scipy.io.wavfile as wavfile


def generate_human_like_voice(duration: float = 3.0, sr: int = 16000) -> np.ndarray:
    """
    Simulates human speech physics:
    - Fundamental frequency F0 ~ 100-220 Hz with natural human jitter and vibrato (micro-tremor).
    - Formants at F1=600Hz, F2=1700Hz, F3=2800Hz (vowel /a/ or /e/).
    - Smooth natural envelope.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Natural pitch variation (vibrato 5Hz + micro-jitter noise)
    base_f0 = np.random.uniform(110.0, 200.0)
    vibrato = 3.0 * np.sin(2 * np.pi * 5.0 * t)
    jitter = np.cumsum(np.random.normal(0, 0.05, size=len(t)))
    f0_t = base_f0 + vibrato + jitter
    phase = np.cumsum(2 * np.pi * f0_t / sr)

    # Glottal source: rich harmonics with 1/f roll-off
    source = np.zeros_like(t)
    for harmonic in range(1, 15):
        amp = 1.0 / (harmonic ** 1.1)
        source += amp * np.sin(harmonic * phase)

    # Vocal tract resonance (formant filtering)
    # Formants: F1 ~ 700Hz, F2 ~ 1800Hz, F3 ~ 2800Hz
    formants = [(700, 100), (1800, 150), (2800, 200)]
    speech = np.zeros_like(source)
    for freq, bw in formants:
        b_freq = freq / (sr / 2)
        b_bw = bw / (sr / 2)
        # Resonant filter
        from scipy.signal import iirpeak
        b, a = iirpeak(b_freq, freq / bw)
        speech += scipy.signal.lfilter(b, a, source)

    # Speech envelope (words/syllables burst structure)
    syllable_rate = np.random.uniform(3.0, 4.5)  # 3-4 syllables per second
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * syllable_rate * t)) ** 2
    # Add subtle breathing noise
    breath = np.random.normal(0, 0.01, size=len(t))
    output = (speech * envelope) + breath

    # Normalize to [-0.85, 0.85]
    peak = np.max(np.abs(output)) + 1e-8
    output = 0.85 * (output / peak)
    return output.astype(np.float32)


def generate_ai_vocoder_voice(duration: float = 3.0, sr: int = 16000) -> np.ndarray:
    """
    Simulates AI neural vocoder artifacts:
    - Overly rigid/quantized pitch track (robotic pitch stability without human micro-tremor).
    - High-frequency phase mismatch & sub-band metallic ringing (typical of HiFi-GAN / MelGAN).
    - Spectral discontinuities across STFT frame boundaries.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Overly rigid pitch (no natural human vibrato or micro-jitter)
    fixed_f0 = np.random.uniform(120.0, 190.0)
    phase = 2 * np.pi * fixed_f0 * t

    # Harmonics
    signal = np.zeros_like(t)
    for h in range(1, 18):
        signal += (1.0 / (h ** 0.9)) * np.sin(h * phase)

    # AI Vocoder Artifact 1: High frequency spectral buzzing / metallic artifacts (4kHz - 7.5kHz)
    vocoder_buzz = 0.08 * np.sin(2 * np.pi * np.random.uniform(4500, 7500) * t)
    
    # AI Vocoder Artifact 2: Frame-boundary periodic phase discontinuity (every 256 samples)
    frame_glitch = np.zeros_like(t)
    hop = 256
    for i in range(0, len(t), hop):
        frame_glitch[i : min(i + 4, len(t))] = np.random.uniform(-0.15, 0.15)

    # Combine
    syllable_rate = np.random.uniform(3.0, 4.0)
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * syllable_rate * t))
    output = (signal * envelope) + vocoder_buzz + frame_glitch

    # Normalize
    peak = np.max(np.abs(output)) + 1e-8
    output = 0.85 * (output / peak)
    return output.astype(np.float32)


def generate_dataset(
    output_root: str = "data/dummy",
    n_train: int = 30,
    n_val: int = 10,
    n_test: int = 10,
    sr: int = 16000
):
    print(f"Generating synthetic verification dataset at: {output_root}")
    splits = {
        "train": n_train,
        "val": n_val,
        "test": n_test
    }

    for split_name, count in splits.items():
        bonafide_dir = os.path.join(output_root, split_name, "bonafide")
        spoof_dir = os.path.join(output_root, split_name, "spoof")
        os.makedirs(bonafide_dir, exist_ok=True)
        os.makedirs(spoof_dir, exist_ok=True)

        print(f"Creating {count} bonafide & {count} spoof samples for split: '{split_name}'...")
        for i in range(count):
            # Generate human sample
            audio_human = generate_human_like_voice(duration=3.0, sr=sr)
            human_path = os.path.join(bonafide_dir, f"human_{i:04d}.wav")
            wavfile.write(human_path, sr, (audio_human * 32767).astype(np.int16))

            # Generate AI sample
            audio_ai = generate_ai_vocoder_voice(duration=3.0, sr=sr)
            ai_path = os.path.join(spoof_dir, f"ai_spoof_{i:04d}.wav")
            wavfile.write(ai_path, sr, (audio_ai * 32767).astype(np.int16))

    print("\nDataset generation finished successfully!")
    print(f"Structure created at '{output_root}':")
    print("  train/ -> bonafide/ (human) & spoof/ (ai)")
    print("  val/   -> bonafide/ (human) & spoof/ (ai)")
    print("  test/  -> bonafide/ (human) & spoof/ (ai)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic voice verification dataset")
    parser.add_argument("--output", type=str, default="data/dummy", help="Output path")
    parser.add_argument("--train_count", type=int, default=30, help="Number of samples per class in train")
    parser.add_argument("--val_count", type=int, default=10, help="Number of samples per class in val")
    parser.add_argument("--test_count", type=int, default=10, help="Number of samples per class in test")

    args = parser.parse_args()
    generate_dataset(
        output_root=args.output,
        n_train=args.train_count,
        n_val=args.val_count,
        n_test=args.test_count
    )
