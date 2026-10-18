"""
Spectro-Temporal CNN + BiLSTM with Self-Attention for Voice Anti-Spoofing.
Combines 2D spectral convolution (capturing vocoder artifacts across frequencies)
with Bidirectional LSTM (capturing unnatural temporal prosody and phase transitions).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSelfAttention(nn.Module):
    """
    Self-attention pooling over the temporal sequence.
    Learns to weigh the most suspicious synthetic frames across an utterance.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Time, Hidden)
        weights = self.attn(x)  # (Batch, Time, 1)
        weights = F.softmax(weights, dim=1)
        context = torch.sum(x * weights, dim=1)  # (Batch, Hidden)
        return context


class SpectroTemporalCNNLSTM(nn.Module):
    """
    Input: (Batch, Channels=3, Freq_Bins, Time_Frames)
    1. 2D CNN layers compress frequency while preserving temporal frames.
    2. Reshape to (Batch, Time, Features).
    3. Bidirectional LSTM models temporal dynamics and vocoder consistency.
    4. Temporal Attention Pooling focuses on glitchy/synthetic bursts.
    5. Linear classifier -> (Human vs AI).
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        # Conv block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1))  # Pool frequency by 2, keep time
        )

        # Conv block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

        # Conv block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, None))  # Fix frequency to 8 bins, arbitrary time
        )

        # CNN output feature dimension = 128 channels * 8 freq bins = 1024
        self.feature_dim = 128 * 8

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # Attention over time
        self.attention = TemporalSelfAttention(hidden_dim=lstm_hidden * 2)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # (B, 128, 8, T)

        b, c, f, t = x.shape
        # Permute to (B, T, C * F)
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (B, T, 2 * hidden)

        # Attention pooling
        attended = self.attention(lstm_out)  # (B, 2 * hidden)

        # Logits
        logits = self.classifier(attended)
        return logits
