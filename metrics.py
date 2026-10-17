"""
Evaluation Metrics for Voice Anti-Spoofing
Includes:
- Equal Error Rate (EER) - The primary benchmark metric of ASVspoof
- Receiver Operating Characteristic (ROC-AUC)
- False Alarm Rate (FAR) & False Rejection Rate (FRR)
- Standard Classification Metrics (Accuracy, Precision, Recall, F1)
"""

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, Tuple, Any


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and the optimal operating threshold.
    labels: Binary array (0 = Bonafide/Human, 1 = Spoof/AI)
    scores: Predicted probability or score of being SPOOF (Class 1)
    
    Returns:
        eer: float (percentage or fraction, e.g. 0.032 -> 3.2% error)
        threshold: float (operating threshold at EER)
    """
    # FPR is False Alarm (Human classified as Spoof)
    # TPR is True Positive Rate (Spoof correctly classified as Spoof)
    # FNR is False Rejection (Spoof missed, classified as Human) = 1 - TPR
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    # Find the index where FPR and FNR are closest
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    optimal_threshold = thresholds[idx]

    return float(eer), float(optimal_threshold)


def evaluate_predictions(labels: np.ndarray, spoof_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Comprehensive anti-spoofing performance evaluation.
    """
    preds = (spoof_probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    
    try:
        auc = roc_auc_score(labels, spoof_probs)
    except Exception:
        auc = 0.5

    eer, opt_thresh = compute_eer(labels, spoof_probs)
    cm = confusion_matrix(labels, preds)

    # cm structure:
    # [[TN (Human as Human), FP (Human as Spoof)],
    #  [FN (Spoof as Human), TP (Spoof as Spoof)]]
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    far = fp / (tn + fp) if (tn + fp) > 0 else 0.0  # False Alarm Rate (Human rejected)
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # Missed Spoof (Vishing attack succeeded!)

    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": float(auc),
        "eer": float(eer),
        "optimal_threshold": float(opt_thresh),
        "current_threshold": float(threshold),
        "false_alarm_rate": float(far),
        "missed_spoof_rate": float(frr),
        "confusion_matrix": cm.tolist(),
        "total_samples": int(len(labels)),
        "bonafide_count": int(np.sum(labels == 0)),
        "spoof_count": int(np.sum(labels == 1))
    }
