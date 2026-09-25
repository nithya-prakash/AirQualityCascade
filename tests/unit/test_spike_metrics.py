import numpy as np
import pytest

from aqcascade.evaluation.spike_metrics import (
    compute_spike_classification_metrics,
    compute_spike_threshold,
)


def test_threshold_is_the_requested_percentile():
    train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    threshold = compute_spike_threshold(train, percentile=90.0)
    assert threshold == pytest.approx(np.percentile(train, 90.0))


def test_perfect_predictions_score_perfectly():
    y = np.array([1.0, 2.0, 20.0, 3.0, 25.0])
    metrics = compute_spike_classification_metrics(y, y, threshold=15.0)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["n_spike_hours_actual"] == 2


def test_missed_spike_lowers_recall_not_precision():
    y_true = np.array([1.0, 20.0, 25.0])  # two real spikes
    y_pred = np.array([1.0, 20.0, 5.0])  # model missed the second spike
    metrics = compute_spike_classification_metrics(y_true, y_pred, threshold=15.0)
    assert metrics["precision"] == 1.0  # every flagged hour really was a spike
    assert metrics["recall"] == 0.5  # only caught 1 of 2 real spikes


def test_pr_auc_uses_continuous_scores_not_thresholded_predictions():
    # Two real spikes (actual concentration >= 15), two calm hours. The
    # model's predicted *scores* rank the spikes correctly, but every
    # predicted value falls just short of the threshold -- so F1 sees zero
    # true positives while PR-AUC (which never thresholds the score) still
    # credits the correct ranking.
    y_true = np.array([1.0, 2.0, 20.0, 25.0])
    y_pred = np.array([1.0, 2.0, 14.9, 14.9])
    metrics = compute_spike_classification_metrics(y_true, y_pred, threshold=15.0)
    assert metrics["f1"] == 0.0  # nothing predicted crosses the threshold
    assert metrics["pr_auc"] == 1.0  # but the ranking itself is perfect
