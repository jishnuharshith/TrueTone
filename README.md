# TrueTone - Anti-Spoofing for Voice-Based Authentication: Defending Against Synthetic Speech Attacks
### IBM Z Datathon 2026 | Real-Time AI Vishing Detection System

---

## 📌 Executive Summary
With the rapid proliferation of generative voice cloning technologies (ElevenLabs, VALL-E, HiFi-GAN, WaveGrad, StyleTTS), financial institutions and consumers face an unprecedented rise in **vishing (voice phishing) attacks**. Attackers synthesize hyper-realistic replicas of bank executives, family members, or automated systems to bypass voice biometrics and trick victims into transferring funds or disclosing OTPs.

This repository provides an enterprise-ready, high-performance **Machine Learning & Deep Learning Voice Anti-Spoofing Pipeline** designed to differentiate between human (bonafide) and synthetic/AI-cloned speech in real-time.

---

## 🏛️ System Architecture

```
Incoming Audio Waveform (.wav, .flac, .mp3, PCM stream)
       │
       ▼
[ Preprocessing & Normalization ]
   ├── Fixed duration windowing (3.0s window / 0.5s hop)
   ├── 16 kHz resampler & DC-offset removal
   └── Telephony & VoIP Bandpass Augmentation (G.711 300Hz-3400Hz)
       │
       ▼
[ Feature Extraction Engine (features.py) ]
   ├── Linear Frequency Cepstral Coefficients (LFCC)
   │     • Linear filter banks retain high-frequency vocoder phase artifacts (0 - 8000 Hz)
   ├── Delta (Velocity) & Delta-Delta (Acceleration) Calculation
   └── Feature Matrix Output: (3 channels, 60 LFCC bins, Time Frames)
       │
       ├───► [ Model Option 1: Squeeze-and-Excitation ResNet (SE-ResNet) ]
       │        • Adaptive channel recalibration highlights anomalous spectral bands
       │        • Dual Global Average + Max spatial pooling
       │
       ├───► [ Model Option 2: Spectro-Temporal CNN-BiLSTM with Self-Attention ]
       │        • 2D CNN extracts local vocoder cues; BiLSTM models unnatural prosody transitions
       │
       └───► [ Model Option 3: Fast Tabular Baseline (XGBoost / Random Forest) ]
                • Instant inference for resource-constrained edge / IoT telecom hardware
       │
       ▼
[ Decision & Anti-Vishing Engine ]
   ├── Calibrated Equal Error Rate (EER) Thresholding
   ├── Confidence Score (Human % vs AI %)
   ├── Latency Tracker (< 20ms real-time ready)
   └── Risk Level Classification (SAFE | SUSPICIOUS | HIGH | CRITICAL)
```

---

## 📂 Project Structure

```
IBM ML/
├── config.py                 # Global hyperparameters & audio configurations
├── features.py               # LFCC, Mel Spectrogram, and Tabular feature extraction
├── dataset.py                # Dataset loaders (directory format & ASVspoof CSV/TXT protocols)
├── models/
│   ├── __init__.py
│   ├── resnet_spec.py        # Squeeze-and-Excitation ResNet audio architecture
│   └── cnn_lstm.py           # Spectro-Temporal CNN + BiLSTM + Attention architecture
├── train.py                  # Training pipeline with Focal Loss, Cosine LR, & EER tracking
├── evaluate.py               # Benchmark script computing EER, ROC-AUC, & Confusion Matrix
├── predict.py                # Single-file / batch inference API & CLI with vishing advisories
├── realtime_detector.py      # Real-time streaming vishing call monitor
├── fast_baseline.py          # Fast interpretable XGBoost baseline on acoustic features
├── generate_dummy_data.py    # Test data generator (human glottal vibrato vs AI vocoder buzz)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Pipeline with Synthetic Data
Before loading large gigabyte datasets, test the entire end-to-end pipeline in seconds:
```bash
# Generate mini verification dataset
python generate_dummy_data.py --output data/dummy

# Train SE-ResNet for 5 verification epochs
python train.py --data_dir data/dummy/train --val_dir data/dummy/val --epochs 5 --model resnet

# Evaluate benchmark metrics (EER, ROC curve)
python evaluate.py --checkpoint checkpoints/best_model.pt --test_dir data/dummy/test

# Test single audio file inference
python predict.py --audio data/dummy/test/spoof/ai_spoof_0001.wav --checkpoint checkpoints/best_model.pt
```

---

## 📊 Training with Datathon Datasets

Once the datasets (**ASVspoof 2019/2021, Fake-or-Real, VoxCeleb, ElevenLabs**) are downloaded, arrange them in either of two standard formats:

### Option A: Subdirectory Structure
```
data/
  train/
    bonafide/   # Human audio files (.wav, .flac)
    spoof/      # AI synthesized / cloned audio files (.wav, .flac)
  val/
    bonafide/
    spoof/
```
Run training:
```bash
python train.py --data_dir data/train --val_dir data/val --model resnet --epochs 25 --batch_size 32
```

### Option B: ASVspoof Protocol File / CSV
```
data/
  audio/
    LA_0001.wav
    LA_0002.wav
  protocol.txt  # Standard ASVspoof format: [speaker_id] [audio_id] [system_id] [key: bonafide/spoof]
```
Run training:
```bash
python train.py --data_dir data/audio --protocol data/protocol.txt --model resnet --epochs 25
```

---

## 📈 Evaluation & Benchmarks
In voice biometric anti-spoofing, the gold-standard metric is **Equal Error Rate (EER)**:
- **EER**: Operating threshold where False Alarm Rate (legitimate human rejected) equals Missed Spoof Rate (AI attack accepted).
- **ROC-AUC**: Overall discriminatory ability across all operating thresholds.

To generate an evaluation report and ROC plot:
```bash
python evaluate.py --checkpoint checkpoints/best_model.pt --test_dir data/test --output_dir eval_results
```

---

## 🛡️ Real-Time Live Call Vishing Defense Monitor

TrueTone monitors live ongoing telephone calls in real time directly from your hardware or via WebSocket:

### 1. Live Call Listening via Hardware (Speakerphone, Headset, or Softphone)
Listen directly to an active live telephone call as it happens (sub-40ms latency, zero recording to disk):

```bash
# A. Speakerphone / Desk Phone / Headset Call:
# Put your mobile on speaker next to your PC mic, or talk into your headset
python realtime_detector.py --live

# B. Desktop Call Software (Zoom, MS Teams, WhatsApp Desktop, Discord, Google Meet):
# Intercept incoming audio directly through Stereo Mix without any external cables!
python realtime_detector.py --list-devices
# Note the device ID for 'Stereo Mix' (e.g. 17) and run:
python realtime_detector.py --live --device 17
```

Live Console Output:
```
================================================================================
  🛡️  TRUETONE: REAL-TIME LIVE CALL ANTI-VISHING DEFENSE MONITOR
  Hardware Input Device:  [17] Stereo Mix (Realtek HD Audio Stereo input)
  Sampling Frequency:     16000 Hz | Window: 3.0s | Hop: 0.5s | Device: CPU
================================================================================
[00:04] | [✅ AUTHENTIC HUMAN  ] | AI Risk: [░░░░░░░░░░░░░░░░░░░░]   0.0% | Latency: 37.1ms
[00:08] | [🚨 CRITICAL VISHING! 🚨] | AI Risk: [████████████████████]  99.9% | Latency: 35.8ms
```
*(On Windows, an audible alarm warning tone automatically sounds via `winsound.Beep` when an AI voice attack is detected!)*

### 2. Headless Streaming API for App Integration (WebSocket & REST)
Launch the lightweight backend service for your mobile app, desktop app, or phone system:
```bash
python realtime_detector.py --api --port 8000
```
- **Real-Time WebSocket Stream**: Connect your app to `ws://localhost:8000/ws/live-call` and stream raw audio buffers (Float32 or Int16 PCM) as the call proceeds.
- **REST Chunk Predictor**: Post base64 audio frames directly to `POST http://localhost:8000/api/predict_chunk`.
- **Health Check**: `GET http://localhost:8000/health`.

### 3. Direct In-Process Python App Integration
You can import the detector directly into your Python / desktop app backend without any HTTP overhead:

```python
from realtime_detector import RealtimeVishingMonitor

# Initialize live monitor (loads SE-ResNet model + adaptive VAD)
monitor = RealtimeVishingMonitor(checkpoint_path="checkpoints/best_model.pt")

# In your app audio stream callback (incoming chunk e.g. 100-200ms):
result = monitor.process_chunk(audio_chunk_numpy)

if result["is_silence"]:
    print("Waiting for caller voice...")
elif result["is_ai"]:
    print(f"🚨 ALERT: AI voice clone detected! Risk: {result['ai_confidence_pct']}%")
else:
    print(f"✅ Authentic human caller. Risk: {result['ai_confidence_pct']}%")
```

### 4. Telephony Trunk & Twilio Live Media Streaming
TrueTone accepts live telephony streams directly:
- **Twilio Media Streams**: Automatically unpacks `{"event": "media", ...}`, decodes 8kHz ITU-T G.711 $\mu$-law, and resamples to 16kHz for continuous sub-40ms threat scoring.
- **Raw PCM Streams**: Accepts 16-bit PCM or 32-bit Float PCM binary chunks directly from softphones, PBX trunks, or Asterisk.

### 5. Audio File Call Simulation (Offline Test)
Simulate an incoming telephone call using a pre-recorded audio file:
```bash
python realtime_detector.py --stream_file data/dataset_test_samples/spoof/LJ001-0004_generated.wav
```

---

## 💡 Python API Integration

Easily integrate the detector into any backend (FastAPI, Flask, WebSocket, or mobile app):

```python
from predict import VoiceSpoofDetector

# Load pre-trained detector
detector = VoiceSpoofDetector(checkpoint_path="checkpoints/best_model.pt")

# Predict on file
result = detector.predict_file("incoming_call.wav")

if result["is_ai"]:
    print(f"ALERT: {result['vishing_warning']}")
    print(f"Confidence: {result['ai_confidence_pct']}% | Latency: {result['latency_ms']} ms")
else:
    print("Caller voice verified as authentic human speech.")
```

---

## 👥 Target Audience & Impact (IBM Z Datathon 2026)
1. **Banking & Financial Customers**: Instant alerts on calls demanding OTPs, wire transfers, or credential verification.
2. **Senior Citizens**: Proactive protection against grandparent scams and voice impersonation.
3. **Enterprises & Employees**: Prevention of CEO fraud and business email/phone compromise.
4. **Telecom & VoIP Providers**: Carrier-grade edge anti-spoofing gateway filtering.
