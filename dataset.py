"""
PyTorch Dataset and DataLoader for Voice Anti-Spoofing
Supports:
1. Directory structure (e.g. data/bonafide/*.wav vs data/spoof/*.wav)
2. ASVspoof protocol / CSV metadata files (audio_path, label)
3. Telephony bandpass filtering (VoIP / vishing simulation)
4. Audio augmentations (additive noise, time-stretching, SpecAugment)
"""

import os
import io
import glob
import random
import numpy as np
import scipy.signal
import scipy.io.wavfile as wavfile
import pandas as pd
from typing import List, Tuple, Optional, Dict, Union
import torch
from torch.utils.data import Dataset, DataLoader

from config import AudioConfig, AugmentationConfig, LABEL_BONAFIDE, LABEL_SPOOF
from features import extract_features


def load_audio_file(filepath: Union[str, bytes], target_sr: int = 16000) -> np.ndarray:
    """
    Robust audio file reader. Tries soundfile -> scipy.io.wavfile -> torchaudio.
    Supports both file path strings and raw audio byte streams.
    Returns mono float32 waveform scaled between [-1.0, 1.0].
    """
    audio = None
    sr = None

    # Handle raw bytes from Arrow / in-memory sources
    if isinstance(filepath, bytes):
        try:
            import soundfile as sf
            audio, sr = sf.read(io.BytesIO(filepath), dtype="float32")
        except Exception:
            pass

        if audio is None:
            try:
                sr, raw = wavfile.read(io.BytesIO(filepath))
                if raw.dtype == np.int16:
                    audio = raw.astype(np.float32) / 32768.0
                elif raw.dtype == np.int32:
                    audio = raw.astype(np.float32) / 2147483648.0
                else:
                    audio = raw.astype(np.float32)
            except Exception as e:
                raise RuntimeError(f"Could not load audio from bytes: {e}")

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        if sr != target_sr:
            num_resampled = int(len(audio) * target_sr / sr)
            audio = scipy.signal.resample(audio, num_resampled)
        return audio.astype(np.float32)

    # Try soundfile
    try:
        import soundfile as sf
        audio, sr = sf.read(filepath, dtype="float32")
    except Exception:
        pass

    # Fallback to scipy for .wav
    if audio is None and str(filepath).lower().endswith(".wav"):
        try:
            sr, raw = wavfile.read(filepath)
            if raw.dtype == np.int16:
                audio = raw.astype(np.float32) / 32768.0
            elif raw.dtype == np.int32:
                audio = raw.astype(np.float32) / 2147483648.0
            else:
                audio = raw.astype(np.float32)
        except Exception:
            pass

    # Fallback to torchaudio
    if audio is None:
        try:
            import torchaudio
            sig, sr = torchaudio.load(filepath)
            audio = sig.squeeze(0).numpy()
        except Exception as e:
            raise RuntimeError(f"Could not load audio file {filepath}: {e}")

    # Convert to mono if multi-channel
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    # Resample if needed
    if sr != target_sr:
        num_resampled = int(len(audio) * target_sr / sr)
        audio = scipy.signal.resample(audio, num_resampled)

    return audio.astype(np.float32)


def pad_or_trim(audio: np.ndarray, target_length: int, random_crop: bool = False) -> np.ndarray:
    """
    Pad with repetition or trim to fixed length.
    """
    length = len(audio)
    if length == target_length:
        return audio
    elif length > target_length:
        if random_crop:
            start = random.randint(0, length - target_length)
        else:
            start = (length - target_length) // 2
        return audio[start : start + target_length]
    else:
        # Repeat signal to fill window
        repeats = (target_length // length) + 1
        extended = np.tile(audio, repeats)
        return extended[:target_length]


class AudioAugmenter:
    """
    Realistic voice augmentations targeting Vishing & Telephony scenarios.
    """
    def __init__(self, cfg: AugmentationConfig, sr: int = 16000):
        self.cfg = cfg
        self.sr = sr

    def apply_telephony_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        Simulate standard G.711 / PSTN telephone bandpass filter (300 Hz - 3400 Hz).
        Vishing calls almost always pass through telecom channels!
        """
        lowcut = 300.0
        highcut = 3400.0
        nyq = 0.5 * self.sr
        low = lowcut / nyq
        high = highcut / nyq
        b, a = scipy.signal.butter(4, [low, high], btype="band")
        return scipy.signal.filtfilt(b, a, audio).astype(np.float32)

    def add_gaussian_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Add additive white Gaussian noise at random SNR.
        """
        snr_db = random.uniform(self.cfg.noise_snr_db_low, self.cfg.noise_snr_db_high)
        signal_power = np.mean(audio ** 2) + 1e-12
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), size=len(audio))
        return (audio + noise).astype(np.float32)

    def augment(self, audio: np.ndarray) -> np.ndarray:
        if not self.cfg.enable:
            return audio
        
        # Telephony filter
        if random.random() < self.cfg.telephony_filter_prob:
            audio = self.apply_telephony_filter(audio)

        # Ambient noise
        if random.random() < self.cfg.noise_prob:
            audio = self.add_gaussian_noise(audio)

        return audio


class VoiceAntiSpoofDataset(Dataset):
    """
    Unified Voice Anti-Spoofing Dataset.
    Supports file paths, in-memory audio byte streams, and pre-extracted samples.
    Labels:
        0 = Human (Bonafide)
        1 = AI / Synthetic (Spoof)
    """
    def __init__(
        self,
        samples: List[Union[Tuple[Union[str, bytes], int], Tuple[Union[str, bytes], int, str]]],
        audio_cfg: AudioConfig,
        aug_cfg: Optional[AugmentationConfig] = None,
        is_train: bool = False
    ):
        self.samples = samples
        self.audio_cfg = audio_cfg
        self.is_train = is_train
        self.augmenter = AudioAugmenter(aug_cfg, sr=audio_cfg.sample_rate) if aug_cfg and is_train else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        item = self.samples[idx]
        if len(item) == 3:
            audio_source, label, path_id = item
        else:
            audio_source, label = item
            path_id = str(audio_source) if isinstance(audio_source, str) else f"sample_{idx}"

        try:
            audio = load_audio_file(audio_source, target_sr=self.audio_cfg.sample_rate)
        except Exception as e:
            # Fallback to zero silence if file is corrupt
            print(f"[Warning] Error loading {path_id}: {e}")
            audio = np.zeros(self.audio_cfg.n_samples, dtype=np.float32)

        # Pad or trim to exact duration
        audio = pad_or_trim(audio, self.audio_cfg.n_samples, random_crop=self.is_train)

        # Augmentation during training
        if self.augmenter is not None:
            audio = self.augmenter.augment(audio)

        # Feature extraction (LFCC / Mel)
        feature = extract_features(
            audio,
            sr=self.audio_cfg.sample_rate,
            feature_type=self.audio_cfg.feature_type,
            with_deltas=self.audio_cfg.with_deltas
        )

        feat_tensor = torch.from_numpy(feature).float()  # (Channels, Freq, Time)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return feat_tensor, label_tensor, str(path_id)


def load_arrow_dataset_samples(
    data_source: str,
    max_samples: Optional[int] = 2000
) -> List[Tuple[bytes, int, str]]:
    """
    Discovers and loads balanced samples from Arrow format shards (e.g. CVoiceFake_small).
    """
    try:
        import pyarrow.ipc as ipc
    except ImportError:
        print("[Warning] pyarrow is not installed. Skipping Arrow dataset loading.")
        return []

    arrow_files = sorted(glob.glob(os.path.join(data_source, "*.arrow")))
    if not arrow_files:
        arrow_files = sorted(glob.glob(os.path.join(data_source, "**", "*.arrow"), recursive=True))
    if not arrow_files:
        return []

    print(f"[*] Detected Arrow Dataset with {len(arrow_files)} shards in: {data_source}")
    max_per_class = (max_samples // 2) if (max_samples and max_samples > 0) else float("inf")
    bona_samples: List[Tuple[bytes, int, str]] = []
    spoof_samples: List[Tuple[bytes, int, str]] = []

    for f in arrow_files:
        if len(bona_samples) >= max_per_class and len(spoof_samples) >= max_per_class:
            break
        try:
            reader = ipc.open_stream(f)
            for batch in reader:
                labels = batch.column("label").to_pylist()
                audios = batch.column("audio")
                paths = batch.column("path").to_pylist() if "path" in batch.schema.names else [f"sample_{i}" for i in range(len(labels))]
                for i, l in enumerate(labels):
                    audio_val = audios[i].as_py()
                    audio_bytes = audio_val.get("bytes") if isinstance(audio_val, dict) else None
                    if audio_bytes is None:
                        continue
                    if l == 0 and len(bona_samples) < max_per_class:
                        bona_samples.append((audio_bytes, LABEL_BONAFIDE, paths[i]))
                    elif l == 1 and len(spoof_samples) < max_per_class:
                        spoof_samples.append((audio_bytes, LABEL_SPOOF, paths[i]))
                    if len(bona_samples) >= max_per_class and len(spoof_samples) >= max_per_class:
                        break
                if len(bona_samples) >= max_per_class and len(spoof_samples) >= max_per_class:
                    break
        except Exception as e:
            print(f"[Warning] Error reading arrow shard {f}: {e}")

    samples = bona_samples + spoof_samples
    random.seed(42)
    random.shuffle(samples)
    print(f"[*] Loaded {len(samples)} balanced Arrow samples: {len(bona_samples)} Real (Bonafide), {len(spoof_samples)} AI (Spoof)")
    return samples


def export_sample_wavs_from_arrow(
    data_source: str,
    output_dir: str = "data/dataset_test_samples",
    num_per_class: int = 50
):
    """
    Exports a small balanced set of WAV files from Arrow shards for testing and single-file prediction.
    """
    os.makedirs(os.path.join(output_dir, "bonafide"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "spoof"), exist_ok=True)
    samples = load_arrow_dataset_samples(data_source, max_samples=num_per_class * 2)
    b_idx = 0
    s_idx = 0
    for audio_bytes, label, orig_path in samples:
        safe_name = os.path.basename(orig_path).replace(".mp3", ".wav")
        if not safe_name.endswith(".wav"):
            safe_name += ".wav"
        if label == LABEL_BONAFIDE and b_idx < num_per_class:
            out_file = os.path.join(output_dir, "bonafide", f"human_{b_idx:03d}_{safe_name}")
            with open(out_file, "wb") as f:
                f.write(audio_bytes)
            b_idx += 1
        elif label == LABEL_SPOOF and s_idx < num_per_class:
            out_file = os.path.join(output_dir, "spoof", f"ai_{s_idx:03d}_{safe_name}")
            with open(out_file, "wb") as f:
                f.write(audio_bytes)
            s_idx += 1
        if b_idx >= num_per_class and s_idx >= num_per_class:
            break
    print(f"[*] Exported {b_idx} bonafide and {s_idx} spoof test WAV files to: {output_dir}")


def discover_dataset_samples(
    data_source: str,
    protocol_file: Optional[str] = None,
    max_samples: Optional[int] = 2000
) -> List[Union[Tuple[str, int], Tuple[bytes, int, str]]]:
    """
    Discovers samples either from:
    1. An Arrow dataset directory (e.g. CVoiceFake_small or folder containing .arrow shards)
    2. A folder with subdirectories:
       - 'bonafide', 'human', 'real' -> Label 0
       - 'spoof', 'fake', 'ai', 'synthetic' -> Label 1
    3. A CSV/TSV protocol file:
       - Columns containing 'path' and 'label' or ASVspoof space-separated protocol
    """
    samples: List[Union[Tuple[str, int], Tuple[bytes, int, str]]] = []

    # 1. Check protocol file if specified
    if protocol_file and os.path.exists(protocol_file):
        samples = parse_protocol_file(protocol_file, data_source)
        if len(samples) > 0:
            return samples

    # 2. Check if data_source is a directory
    if os.path.isdir(data_source):
        # Check if wavefake_4s subfolder exists inside data_source
        wavefake_4s_dir = os.path.join(data_source, "wavefake_4s")
        if os.path.isdir(wavefake_4s_dir):
            wf_samples = discover_dataset_samples(wavefake_4s_dir, max_samples=max_samples)
            if len(wf_samples) > 0:
                print(f"[*] Found WaveFake 4s dataset in: {wavefake_4s_dir} ({len(wf_samples)} samples)")
                return wf_samples

        wavefake_dir = os.path.join(data_source, "_sources_wavefake")
        if os.path.isdir(wavefake_dir) and not os.path.basename(data_source).startswith("CVoiceFake"):
            wf_samples = discover_dataset_samples(wavefake_dir, max_samples=max_samples)
            if len(wf_samples) > 0:
                print(f"[*] Found WaveFake dataset in: {wavefake_dir} ({len(wf_samples)} samples)")
                return wf_samples

        # Look for standard subdirectories with audio files
        bonafide_dirs = ["bonafide", "human", "real", "bona_fide"]
        spoof_dirs = ["spoof", "fake", "ai", "synthetic", "cloned"]

        audio_exts = ("*.wav", "*.flac", "*.mp3", "*.m4a", "*.ogg")

        found_structure = False
        for entry in os.listdir(data_source):
            full_sub = os.path.join(data_source, entry)
            if not os.path.isdir(full_sub):
                continue

            entry_lower = entry.lower()
            # Accurately classify based on the primary voice directory name
            if entry_lower.startswith("fake_voice") or entry_lower.startswith("spoof") or entry_lower.startswith("ai") or entry_lower.startswith("synthetic"):
                found_structure = True
                for ext in audio_exts:
                    for f in glob.glob(os.path.join(full_sub, "**", ext), recursive=True):
                        samples.append((f, LABEL_SPOOF))
            elif entry_lower.startswith("real_voice") or entry_lower.startswith("bonafide") or entry_lower.startswith("human") or entry_lower.startswith("bona_fide"):
                found_structure = True
                for ext in audio_exts:
                    for f in glob.glob(os.path.join(full_sub, "**", ext), recursive=True):
                        samples.append((f, LABEL_BONAFIDE))
            elif any(s in entry_lower for s in spoof_dirs) and not any(b in entry_lower for b in bonafide_dirs):
                found_structure = True
                for ext in audio_exts:
                    for f in glob.glob(os.path.join(full_sub, "**", ext), recursive=True):
                        samples.append((f, LABEL_SPOOF))
            elif any(b in entry_lower for b in bonafide_dirs) and not any(s in entry_lower for s in spoof_dirs):
                found_structure = True
                for ext in audio_exts:
                    for f in glob.glob(os.path.join(full_sub, "**", ext), recursive=True):
                        samples.append((f, LABEL_BONAFIDE))

        if found_structure and len(samples) > 0:
            if max_samples and max_samples > 0 and len(samples) > max_samples:
                bona = [s for s in samples if s[1] == LABEL_BONAFIDE]
                spoof = [s for s in samples if s[1] == LABEL_SPOOF]
                n_per = max_samples // 2
                random.seed(42)
                random.shuffle(bona)
                random.shuffle(spoof)
                samples = bona[:n_per] + spoof[:n_per]
                random.shuffle(samples)
            return samples

        # Check for Arrow files (.arrow)
        arrow_samples = load_arrow_dataset_samples(data_source, max_samples=max_samples)
        if len(arrow_samples) > 0:
            return arrow_samples

        if not found_structure:
            # Check for protocol file inside data_source
            for proto_candidate in ["protocol.txt", "protocol.csv", "meta.csv", "metadata.csv"]:
                p_path = os.path.join(data_source, proto_candidate)
                if os.path.exists(p_path):
                    return parse_protocol_file(p_path, data_source)

    return samples


def parse_protocol_file(protocol_path: str, base_audio_dir: str) -> List[Tuple[str, int]]:
    """
    Parses CSV, TSV or standard ASVspoof protocol text files.
    """
    samples: List[Tuple[str, int]] = []
    
    # Try reading as pandas CSV
    try:
        df = pd.read_csv(protocol_path, sep=None, engine="python")
        col_names = [str(c).lower().strip() for c in df.columns]
        
        path_col = None
        label_col = None
        for i, c in enumerate(col_names):
            if c in ("path", "filename", "file", "audio_path", "audio", "utt_id"):
                path_col = df.columns[i]
            elif c in ("label", "target", "class", "is_spoof", "key"):
                label_col = df.columns[i]

        if path_col and label_col:
            for _, row in df.iterrows():
                p = str(row[path_col]).strip()
                if not os.path.isabs(p):
                    p = os.path.join(base_audio_dir, p)
                lbl_val = str(row[label_col]).lower().strip()
                lbl = LABEL_BONAFIDE if lbl_val in ("bonafide", "real", "human", "0", "0.0") else LABEL_SPOOF
                samples.append((p, lbl))
            return samples
    except Exception:
        pass

    # Fallback to standard ASVspoof space-delimited text parsing
    # Standard format: [speaker_id] [audio_id] [system_id] [key: bonafide/spoof]
    with open(protocol_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                # Key is often the last token
                key = parts[-1].lower()
                audio_id = parts[1] if len(parts) >= 4 else parts[0]
                
                # Check if audio_id has extension, else add .wav / .flac
                matched_path = None
                for ext in ["", ".wav", ".flac", ".mp3"]:
                    candidate = os.path.join(base_audio_dir, audio_id + ext)
                    if os.path.exists(candidate):
                        matched_path = candidate
                        break
                
                if matched_path:
                    lbl = LABEL_BONAFIDE if "bonafide" in key else LABEL_SPOOF
                    samples.append((matched_path, lbl))

    return samples


def create_dataloaders(
    train_samples: List[Tuple[str, int]],
    val_samples: Optional[List[Tuple[str, int]]] = None,
    audio_cfg: Optional[AudioConfig] = None,
    aug_cfg: Optional[AugmentationConfig] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    val_split_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train and validation PyTorch DataLoaders.
    If val_samples is not provided, automatically splits train_samples.
    """
    if audio_cfg is None:
        audio_cfg = AudioConfig()
    if aug_cfg is None:
        aug_cfg = AugmentationConfig()

    if val_samples is None or len(val_samples) == 0:
        # Split train_samples
        random.seed(seed)
        shuffled = list(train_samples)
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * (1.0 - val_split_ratio))
        train_samples = shuffled[:split_idx]
        val_samples = shuffled[split_idx:]

    train_ds = VoiceAntiSpoofDataset(train_samples, audio_cfg=audio_cfg, aug_cfg=aug_cfg, is_train=True)
    val_ds = VoiceAntiSpoofDataset(val_samples, audio_cfg=audio_cfg, aug_cfg=None, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader
