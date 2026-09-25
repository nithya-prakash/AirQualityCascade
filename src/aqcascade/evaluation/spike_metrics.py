"""Pollution-spike classification metrics -- an optional supplement to the
regression metrics, closer to what an early-warning system actually needs:
not "how close was the number" but "did we correctly flag the dangerous
hours".

Threshold policy: a fixed regulatory concentration (e.g. a 24h WHO/EU
guideline) doesn't directly apply here -- those are defined for 24-hour
averages, not the single-hour PM2.5 readings this project forecasts. Using
one anyway would misrepresent what the number means. Instead the spike
threshold is a train-period statistical percentile (e.g. the 90th), which
adapts to the actual scale of the data and is definitionally not fit using
any information from the test period.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


def compute_spike_threshold(train_values: np.ndarray, percentile: float = 90.0) -> float:
    return float(np.percentile(train_values, percentile))


def compute_spike_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float
) -> dict:
    """Precision/recall/F1 from thresholding both true and predicted values
    at `threshold`; PR-AUC from the continuous predicted value as a score
    (never thresholded), which is the metric's whole point -- it summarizes
    ranking quality across every possible threshold, not just this one.
    """
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)

    return {
        "threshold": threshold,
        "n_spike_hours_actual": int(y_true_binary.sum()),
        "n_spike_hours_predicted": int(y_pred_binary.sum()),
        "precision": float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
        "recall": float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
        "f1": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true_binary, y_pred)),
    }
