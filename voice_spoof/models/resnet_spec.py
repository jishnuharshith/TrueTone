"""
Squeeze-and-Excitation ResNet (SE-ResNet) for Voice Anti-Spoofing
Adapted for 2D Audio Feature Maps (LFCC / Mel Spectrogram with Deltas).
Channel attention adaptively highlights vocoder artifact frequency bands.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for adaptive channel recalibration.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class SEResidualBlock(nn.Module):
    """
    Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> SEBlock + Shortcut
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.relu(out + residual)
        return out


class SqueezeExcitationResNet(nn.Module):
    """
    ResNet-SE Architecture optimized for Audio Anti-Spoofing.
    Input shape: (Batch, in_channels=3, Freq_Bins, Time_Frames)
    Output shape: (Batch, num_classes=2) -> [P(Human), P(AI)]
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_channels: int = 32,
        dropout: float = 0.3
    ):
        super().__init__()
        # Initial stem conv
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 4 Residual Stages
        self.stage1 = self._make_stage(base_channels, base_channels, num_blocks=2, stride=1)
        self.stage2 = self._make_stage(base_channels, base_channels * 2, num_blocks=2, stride=2)
        self.stage3 = self._make_stage(base_channels * 2, base_channels * 4, num_blocks=2, stride=2)
        self.stage4 = self._make_stage(base_channels * 4, base_channels * 8, num_blocks=2, stride=2)

        # Dual pooling: concatenation of Global Average Pooling & Global Max Pooling
        final_channels = base_channels * 8
        self.classifier = nn.Sequential(
            nn.Linear(final_channels * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def _make_stage(self, in_ch: int, out_ch: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = [SEResidualBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(SEResidualBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Global Avg + Max Pool across spatial dimensions (F, T)
        avg_pool = torch.mean(x, dim=[2, 3])
        max_pool = torch.amax(x, dim=[2, 3])
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2 * final_channels)

        logits = self.classifier(pooled)
        return logits


def get_resnet_antispoof(in_channels: int = 3, num_classes: int = 2) -> SqueezeExcitationResNet:
    return SqueezeExcitationResNet(in_channels=in_channels, num_classes=num_classes)
