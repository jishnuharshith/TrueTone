"""
Voice Anti-Spoofing Deep Learning Architectures
"""
from .resnet_spec import SqueezeExcitationResNet, get_resnet_antispoof
from .cnn_lstm import SpectroTemporalCNNLSTM

__all__ = ["SqueezeExcitationResNet", "get_resnet_antispoof", "SpectroTemporalCNNLSTM"]
