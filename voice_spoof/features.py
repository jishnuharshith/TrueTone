"""
Acoustic Feature Extraction for Voice Anti-Spoofing
Supports:
1. LFCC (Linear Frequency Cepstral Coefficients) + Deltas + Delta-Deltas (Gold standard in ASVspoof)
2. Log-Mel Spectrograms
3. Tabular acoustic features for fast ML baselines (XGBoost, RF)
"""

import numpy as np
import scipy.signal
import scipy.fftpack
from typing import Tuple, Optional

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def compute_deltas_numpy(x: np.ndarray, width: int = 5) -> np.ndarray:
    """
    Compute delta (velocity) or delta-delta (acceleration) coefficients along the time axis (last axis).
    """
    half_w = width // 2
    padded = np.pad(x, ((0, 0), (half_w, half_w)), mode="edge")
    kernel = np.arange(-half_w, half_w + 1)
    denom = 2 * sum(i ** 2 for i in range(1, half_w + 1))
    
    deltas = np.zeros_like(x)
    for i in range(x.shape[1]):
        window = padded[:, i : i + width]
        deltas[:, i] = np.sum(window * kernel, axis=1) / denom
    return deltas


def extract_lfcc_numpy(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_filterbands: int = 70,
    n_lfcc: int = 60,
    with_deltas: bool = True
) -> np.ndarray:
    """
    Extract Linear Frequency Cepstral Coefficients (LFCC) using NumPy and SciPy.
    Linear filter banks preserve high-frequency artifacts left by AI neural vocoders.
    Output shape:
        If with_deltas: (3, n_lfcc, time_steps) [Static, Delta, Delta-Delta]
        Else: (1, n_lfcc, time_steps)
    """
    # 1. Pre-emphasis filter
    audio_pre = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # 2. Short-time Fourier Transform (STFT)
    window = np.hamming(win_length)
    frames = []
    for start in range(0, len(audio_pre) - win_length + 1, hop_length):
        frames.append(audio_pre[start : start + win_length] * window)
    
    if len(frames) == 0:
        frames = [np.zeros(win_length)]
    frames = np.array(frames)

    # Magnitude spectrum
    mag_spec = np.abs(np.fft.rfft(frames, n=n_fft))  # Shape: (T, n_fft//2 + 1)

    # 3. Linear Filter Bank creation
    num_bins = n_fft // 2 + 1
    lin_points = np.linspace(0, sr / 2, n_filterbands + 2)
    bin_points = np.floor((n_fft + 1) * lin_points / sr).astype(int)

    fbank = np.zeros((n_filterbands, num_bins))
    for m in range(1, n_filterbands + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    # 4. Filter bank energies
    filter_energies = np.dot(mag_spec, fbank.T)
    filter_energies = np.where(filter_energies == 0, np.finfo(float).eps, filter_energies)
    log_filter_energies = 10 * np.log10(filter_energies)

    # 5. Discrete Cosine Transform (DCT) -> LFCC
    lfcc = scipy.fftpack.dct(log_filter_energies, type=2, axis=1, norm="ortho")[:, :n_lfcc].T  # (n_lfcc, T)

    # CMVN (Cepstral Mean and Variance Normalization) per-feature
    lfcc_mean = np.mean(lfcc, axis=1, keepdims=True)
    lfcc_std = np.std(lfcc, axis=1, keepdims=True) + 1e-8
    lfcc = (lfcc - lfcc_mean) / lfcc_std

    if with_deltas:
        d1 = compute_deltas_numpy(lfcc)
        d2 = compute_deltas_numpy(d1)
        # Normalize delta channels
        d1 = (d1 - np.mean(d1, axis=1, keepdims=True)) / (np.std(d1, axis=1, keepdims=True) + 1e-8)
        d2 = (d2 - np.mean(d2, axis=1, keepdims=True)) / (np.std(d2, axis=1, keepdims=True) + 1e-8)
        # Shape: (3, n_lfcc, T) -> ideal for 3-channel Conv2D / ResNet
        feature = np.stack([lfcc, d1, d2], axis=0)
    else:
        feature = lfcc[np.newaxis, :, :]

    return feature.astype(np.float32)


def extract_mel_numpy(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    with_deltas: bool = True
) -> np.ndarray:
    """
    Extract Log-Mel Spectrogram with Deltas using NumPy/SciPy.
    Output shape: (3, n_mels, time_steps) if with_deltas else (1, n_mels, time_steps).
    """
    # STFT
    window = np.hanning(win_length)
    frames = []
    for start in range(0, len(audio) - win_length + 1, hop_length):
        frames.append(audio[start : start + win_length] * window)
    if len(frames) == 0:
        frames = [np.zeros(win_length)]
    frames = np.array(frames)
    power_spec = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2

    # Mel filter bank
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]

        for k in range(f_left, f_center):
            if f_center != f_left:
                fbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right != f_center:
                fbank[m - 1, k] = (f_right - k) / (f_right - f_center)

    mel_spec = np.dot(power_spec, fbank.T)
    mel_spec = np.where(mel_spec == 0, np.finfo(float).eps, mel_spec)
    log_mel = 10.0 * np.log10(mel_spec).T  # (n_mels, T)

    # Normalize
    mean = np.mean(log_mel)
    std = np.std(log_mel) + 1e-8
    log_mel = (log_mel - mean) / std

    if with_deltas:
        d1 = compute_deltas_numpy(log_mel)
        d2 = compute_deltas_numpy(d1)
        feature = np.stack([log_mel, d1, d2], axis=0)
    else:
        feature = log_mel[np.newaxis, :, :]

    return feature.astype(np.float32)


def extract_features(
    audio: np.ndarray,
    sr: int = 16000,
    feature_type: str = "lfcc",
    with_deltas: bool = True
) -> np.ndarray:
    """
    Unified feature extractor function for audio signals.
    """
    if feature_type.lower() == "lfcc":
        return extract_lfcc_numpy(audio, sr=sr, with_deltas=with_deltas)
    elif feature_type.lower() in ("mel", "logmel"):
        return extract_mel_numpy(audio, sr=sr, with_deltas=with_deltas)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}. Use 'lfcc' or 'mel'.")


def extract_tabular_acoustic_features(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Extract a compact 1D vector of acoustic & statistical features suitable for
    fast classifiers (XGBoost, Random Forest, SVM) and edge/embedded deployment.
    Features include:
    - LFCC stats (mean, std, skew across time)
    - Spectral Centroid, Bandwidth, Rolloff, Flatness
    - Zero Crossing Rate (ZCR)
    - Root Mean Square Energy (RMS)
    """
    # 1. Base LFCC
    lfcc_feat = extract_lfcc_numpy(audio, sr=sr, n_lfcc=30, with_deltas=False)[0]  # (30, T)
    lfcc_mean = np.mean(lfcc_feat, axis=1)
    lfcc_std = np.std(lfcc_feat, axis=1)

    # 2. Time-domain statistics
    zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0
    rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
    peak = np.max(np.abs(audio))
    crest_factor = peak / (rms + 1e-12)

    # 3. Spectral statistics
    mag = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    sum_mag = np.sum(mag) + 1e-12
    
    # Spectral Centroid
    centroid = np.sum(freqs * mag) / sum_mag
    # Spectral Spread / Bandwidth
    spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / sum_mag)
    # Spectral Flatness
    geo_mean = np.exp(np.mean(np.log(mag + 1e-12)))
    arith_mean = np.mean(mag) + 1e-12
    flatness = geo_mean / arith_mean
    # Spectral Rolloff (85% energy)
    cumsum_mag = np.cumsum(mag)
    rolloff_idx = np.searchsorted(cumsum_mag, 0.85 * sum_mag)
    rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]

    # Combine into 1D feature vector
    stats = np.array([zcr, rms, peak, crest_factor, centroid, spread, flatness, rolloff], dtype=np.float32)
    vector = np.concatenate([lfcc_mean, lfcc_std, stats])
    return vector
