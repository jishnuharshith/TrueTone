"""
Fast Interpretable Baseline (XGBoost / Random Forest) on Acoustic Features
IBM Z Datathon 2026
"""

import os
import argparse
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from config import AudioConfig, LABEL_BONAFIDE, LABEL_SPOOF
from dataset import discover_dataset_samples, load_audio_file, pad_or_trim
from features import extract_tabular_acoustic_features
from metrics import evaluate_predictions, compute_eer


def extract_features_from_dataset(samples, audio_cfg: AudioConfig):
    X = []
    y = []
    print(f"Extracting tabular acoustic features for {len(samples)} audio clips...")
    for path, label in samples:
        try:
            audio = load_audio_file(path, target_sr=audio_cfg.sample_rate)
            audio = pad_or_trim(audio, audio_cfg.n_samples, random_crop=False)
            feat = extract_tabular_acoustic_features(audio, sr=audio_cfg.sample_rate)
            X.append(feat)
            y.append(label)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def train_fast_baseline(
    data_dir: str,
    protocol_file: str = None,
    model_type: str = "xgboost",
    save_path: str = "checkpoints/fast_baseline.joblib"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    audio_cfg = AudioConfig()

    samples = discover_dataset_samples(data_dir, protocol_file=protocol_file)
    if len(samples) == 0:
        raise ValueError(f"No audio samples found in: {data_dir}")

    # Random split 80/20
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    split = int(0.8 * len(samples))
    train_samples = [samples[i] for i in indices[:split]]
    val_samples = [samples[i] for i in indices[split:]]

    X_train, y_train = extract_features_from_dataset(train_samples, audio_cfg)
    X_val, y_val = extract_features_from_dataset(val_samples, audio_cfg)

    print(f"\nTraining samples: {len(X_train)} | Validation samples: {len(X_val)}")
    print(f"Feature vector dimensionality: {X_train.shape[1]}")

    if model_type.lower() == "xgboost":
        clf = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            eval_metric="logloss",
            random_state=42
        )
    else:
        clf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)

    print(f"Fitting {model_type.upper()} model...")
    clf.fit(X_train, y_train)

    val_probs = clf.predict_proba(X_val)[:, LABEL_SPOOF]
    metrics = evaluate_predictions(y_val, val_probs)

    print("\n" + "=" * 50)
    print(f"     {model_type.upper()} FAST BASELINE RESULTS")
    print("=" * 50)
    print(f"Accuracy:                 {metrics['accuracy'] * 100:.2f}%")
    print(f"Equal Error Rate (EER):   {metrics['eer'] * 100:.2f}%")
    print(f"ROC-AUC:                  {metrics['roc_auc']:.4f}")
    print(f"Precision:                {metrics['precision'] * 100:.2f}%")
    print(f"Recall:                   {metrics['recall'] * 100:.2f}%")
    print("=" * 50)

    joblib.dump({"model": clf, "type": model_type, "metrics": metrics}, save_path)
    print(f"Saved baseline model to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fast Acoustic Baseline (XGBoost / RF)")
    parser.add_argument("--data_dir", type=str, default="data/train", help="Dataset directory")
    parser.add_argument("--protocol", type=str, default=None, help="Protocol file")
    parser.add_argument("--model", type=str, default="xgboost", choices=["xgboost", "rf"])
    parser.add_argument("--save_path", type=str, default="checkpoints/fast_baseline.joblib")

    args = parser.parse_args()
    train_fast_baseline(args.data_dir, protocol_file=args.protocol, model_type=args.model, save_path=args.save_path)
