"""
Training Pipeline for Voice Anti-Spoofing (Human vs AI Voice)
IBM Z Datathon 2026
"""

import os
import sys
import json
import argparse
import time
from typing import Tuple, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import AudioConfig, AugmentationConfig, TrainConfig, LABEL_BONAFIDE, LABEL_SPOOF
from dataset import discover_dataset_samples, create_dataloaders
from models import get_resnet_antispoof, SpectroTemporalCNNLSTM
from metrics import compute_eer, evaluate_predictions


class FocalLoss(nn.Module):
    """
    Focal Loss to counter class imbalance between real and synthetic voice samples.
    """
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return torch.mean(focal_loss)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: str
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (features, labels, _) in enumerate(loader):
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def validate(
    model: nn.Module,
    loader,
    criterion,
    device: str
) -> Tuple[float, float, float, Dict]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for features, labels, _ in loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            total += len(labels)

            probs = F.softmax(logits, dim=1)
            # Probability of Class 1 (AI / Spoof)
            spoof_prob = probs[:, LABEL_SPOOF].cpu().numpy()

            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(spoof_prob)

    avg_loss = total_loss / max(total, 1)
    labels_arr = np.array(all_labels)
    scores_arr = np.array(all_scores)

    eval_results = evaluate_predictions(labels_arr, scores_arr)
    return avg_loss, eval_results["accuracy"], eval_results["eer"], eval_results


def run_training(
    train_data_source: str,
    val_data_source: str = None,
    protocol_file: str = None,
    model_name: str = "resnet",
    feature_type: str = "lfcc",
    num_epochs: int = 25,
    batch_size: int = 32,
    lr: float = 3e-4,
    save_dir: str = "checkpoints",
    max_samples: int = 2000
):
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Voice Anti-Spoofing Training Pipeline ===")
    print(f"Device: {device}")
    print(f"Model Architecture: {model_name.upper()}")
    print(f"Acoustic Feature: {feature_type.upper()} + Deltas (3 channels)")

    audio_cfg = AudioConfig(feature_type=feature_type)
    aug_cfg = AugmentationConfig(enable=True)

    # 1. Discover training data
    print(f"\n[1/4] Discovering datasets from: {train_data_source}")
    train_samples = discover_dataset_samples(train_data_source, protocol_file=protocol_file, max_samples=max_samples)
    if len(train_samples) == 0:
        raise ValueError(
            f"No valid audio samples found in {train_data_source}!\n"
            "Ensure the folder contains subdirectories 'bonafide' (or 'human') and 'spoof' (or 'ai'), "
            "an Arrow dataset (.arrow shards), or a valid protocol.txt / metadata.csv."
        )

    # Auto-export a small set of test audio files from Arrow shards if applicable
    test_wav_dir = os.path.join(os.path.dirname(__file__), "data", "dataset_test_samples")
    if not os.path.exists(test_wav_dir):
        try:
            from dataset import export_sample_wavs_from_arrow
            export_sample_wavs_from_arrow(train_data_source, output_dir=test_wav_dir, num_per_class=30)
        except Exception as e:
            pass

    val_samples = None
    if val_data_source and os.path.exists(val_data_source):
        print(f"Discovering validation samples from: {val_data_source}")
        val_samples = discover_dataset_samples(val_data_source, max_samples=max_samples // 4 if max_samples else None)

    train_loader, val_loader = create_dataloaders(
        train_samples,
        val_samples=val_samples,
        audio_cfg=audio_cfg,
        aug_cfg=aug_cfg,
        batch_size=batch_size,
        num_workers=0
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # 2. Initialize Model
    print(f"\n[2/4] Initializing model ({model_name})...")
    if model_name.lower() in ("resnet", "se-resnet", "resnet-se"):
        model = get_resnet_antispoof(in_channels=3, num_classes=2)
    elif model_name.lower() in ("cnnlstm", "cnn_lstm"):
        model = SpectroTemporalCNNLSTM(in_channels=3, num_classes=2)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Use 'resnet' or 'cnnlstm'.")

    model = model.to(device)

    # 3. Criterion, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # 4. Training Loop
    print(f"\n[3/4] Commencing Training for {num_epochs} epochs...")
    best_eer = float("inf")
    patience_counter = 0
    patience_limit = 7
    history = []

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_eer, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start_time

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_eer": val_eer,
            "val_auc": val_metrics["roc_auc"],
            "optimal_threshold": val_metrics["optimal_threshold"]
        })

        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | "
            f"Val EER: {val_eer*100:.2f}% | ROC-AUC: {val_metrics['roc_auc']:.4f}"
        )

        # Checkpoint if best EER
        if val_eer < best_eer:
            best_eer = val_eer
            patience_counter = 0
            best_path = os.path.join(save_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_name": model_name,
                "feature_type": feature_type,
                "val_eer": val_eer,
                "val_acc": val_acc,
                "val_auc": val_metrics["roc_auc"],
                "optimal_threshold": val_metrics["optimal_threshold"]
            }, best_path)
            print(f"  >>> Best model saved to {best_path} (EER: {val_eer*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"\n[Early Stopping] No improvement in EER for {patience_limit} epochs.")
                break

    # Save training history
    history_file = os.path.join(save_dir, "training_history.json")
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[4/4] Training Complete!")
    print(f"Best Validation EER: {best_eer*100:.2f}%")
    print(f"Model saved to: {os.path.join(save_dir, 'best_model.pt')}")
    print(f"Training history saved to: {history_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Voice Anti-Spoofing Classifier")
    parser.add_argument("--data_dir", type=str, default="data/train", help="Path to training data directory")
    parser.add_argument("--val_dir", type=str, default=None, help="Path to validation data directory (optional)")
    parser.add_argument("--protocol", type=str, default=None, help="Path to ASVspoof protocol.txt or metadata.csv")
    parser.add_argument("--model", type=str, default="resnet", choices=["resnet", "cnnlstm"], help="Model architecture")
    parser.add_argument("--feature", type=str, default="lfcc", choices=["lfcc", "mel"], help="Acoustic feature type")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--max_samples", type=int, default=2000, help="Maximum balanced samples to load from Arrow dataset (0 for all)")

    args = parser.parse_args()
    run_training(
        train_data_source=args.data_dir,
        val_data_source=args.val_dir,
        protocol_file=args.protocol,
        model_name=args.model,
        feature_type=args.feature,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        max_samples=args.max_samples
    )
