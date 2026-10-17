"""
Evaluation and Benchmarking Script for Voice Anti-Spoofing
IBM Z Datathon 2026
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from config import AudioConfig, LABEL_BONAFIDE, LABEL_SPOOF, LABEL_NAMES
from dataset import discover_dataset_samples, VoiceAntiSpoofDataset
from models import get_resnet_antispoof, SpectroTemporalCNNLSTM
from metrics import evaluate_predictions, compute_eer


def load_trained_model(checkpoint_path: str, device: str = "cpu"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", "resnet")
    feature_type = checkpoint.get("feature_type", "lfcc")

    if model_name.lower() in ("resnet", "se-resnet", "resnet-se"):
        model = get_resnet_antispoof(in_channels=3, num_classes=2)
    elif model_name.lower() in ("cnnlstm", "cnn_lstm"):
        model = SpectroTemporalCNNLSTM(in_channels=3, num_classes=2)
    else:
        raise ValueError(f"Unknown architecture in checkpoint: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    opt_thresh = checkpoint.get("optimal_threshold", 0.5)
    return model, model_name, feature_type, opt_thresh


def evaluate_dataset(
    checkpoint_path: str,
    test_data_dir: str,
    protocol_file: str = None,
    threshold: float = None,
    output_dir: str = "eval_results"
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, model_name, feature_type, saved_thresh = load_trained_model(checkpoint_path, device=device)
    eval_thresh = threshold if threshold is not None else saved_thresh

    audio_cfg = AudioConfig(feature_type=feature_type)
    test_samples = discover_dataset_samples(test_data_dir, protocol_file=protocol_file)

    if len(test_samples) == 0:
        raise ValueError(f"No test audio samples found in: {test_data_dir}")

    test_ds = VoiceAntiSpoofDataset(test_samples, audio_cfg=audio_cfg, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

    print(f"=== Voice Anti-Spoofing Benchmark ===")
    print(f"Model: {model_name.upper()} | Feature: {feature_type.upper()}")
    print(f"Evaluating {len(test_samples)} audio files on {device}...")

    all_labels = []
    all_scores = []
    all_paths = []

    with torch.no_grad():
        for features, labels, paths in test_loader:
            features = features.to(device)
            logits = model(features)
            probs = F.softmax(logits, dim=1)
            spoof_prob = probs[:, LABEL_SPOOF].cpu().numpy()

            all_labels.extend(labels.numpy())
            all_scores.extend(spoof_prob)
            all_paths.extend(paths)

    labels_arr = np.array(all_labels)
    scores_arr = np.array(all_scores)

    results = evaluate_predictions(labels_arr, scores_arr, threshold=eval_thresh)

    print("\n" + "=" * 50)
    print("           BENCHMARK RESULTS REPORT")
    print("=" * 50)
    print(f"Total Test Audio Files:     {results['total_samples']}")
    print(f" - Human (Bonafide):        {results['bonafide_count']}")
    print(f" - AI / Synthetic (Spoof):  {results['spoof_count']}")
    print("-" * 50)
    print(f"Equal Error Rate (EER):     {results['eer'] * 100:.2f}%  (ASVspoof Primary Metric)")
    print(f"ROC-AUC Score:              {results['roc_auc']:.4f}")
    print(f"Overall Accuracy:           {results['accuracy'] * 100:.2f}%")
    print(f"Precision:                  {results['precision'] * 100:.2f}%")
    print(f"Recall (Spoof Detection):   {results['recall'] * 100:.2f}%")
    print(f"F1-Score:                   {results['f1']:.4f}")
    print("-" * 50)
    print(f"Operating Threshold:        {eval_thresh:.4f}")
    print(f"Optimal EER Threshold:      {results['optimal_threshold']:.4f}")
    print(f"False Alarm Rate (FAR):     {results['false_alarm_rate'] * 100:.2f}% (Human falsely flagged)")
    print(f"Missed Spoof Rate (FRR):    {results['missed_spoof_rate'] * 100:.2f}% (Vishing attack slipped through)")
    print("=" * 50)

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "benchmark_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    # Plot ROC Curve
    try:
        fpr, tpr, _ = roc_curve(labels_arr, scores_arr, pos_label=1)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color="crimson", lw=2, label=f"ROC curve (AUC = {results['roc_auc']:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
        plt.scatter([results['eer']], [1.0 - results['eer']], color="black", zorder=5,
                    label=f"EER Operating Point ({results['eer']*100:.2f}%)")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (Human Flagged as AI)")
        plt.ylabel("True Positive Rate (AI Correctly Caught)")
        plt.title("Voice Anti-Spoofing ROC Curve - IBM Z Datathon 2026")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plot_path = os.path.join(output_dir, "roc_curve.png")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"ROC plot saved to: {plot_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")

    print(f"Full metrics saved to: {metrics_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Voice Anti-Spoofing Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--test_dir", type=str, default="data/test", help="Path to test audio directory")
    parser.add_argument("--protocol", type=str, default=None, help="Path to test protocol file")
    parser.add_argument("--threshold", type=float, default=None, help="Classification threshold (default: auto from EER)")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory to save evaluation reports")

    args = parser.parse_args()
    evaluate_dataset(
        checkpoint_path=args.checkpoint,
        test_data_dir=args.test_dir,
        protocol_file=args.protocol,
        threshold=args.threshold,
        output_dir=args.output_dir
    )
