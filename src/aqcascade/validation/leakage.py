"""Automated checks that guard against temporal leakage.

Used both as a one-off assertion when building train/val/test splits
(Phase 5+) and as the basis for the leakage tests in tests/data/.
"""

from __future__ import annotations

import pandas as pd


class LeakageError(ValueError):
    """Raised when a split or feature construction would leak future data."""


def assert_chronological_split(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    time_col: str = "timestamp",
) -> None:
    """Verify train < val < test in time, with no overlap.

    Random splitting is never used for this project's evaluation — every
    split must be chronological so validation/test only ever contain
    periods strictly after what the model was trained on.
    """
    train_max = train[time_col].max()
    val_min, val_max = val[time_col].min(), val[time_col].max()
    test_min = test[time_col].min()

    if train_max >= val_min:
        raise LeakageError(
            f"Train period (up to {train_max}) overlaps validation period "
            f"(starting {val_min}) — splits must be strictly chronological."
        )
    if val_max >= test_min:
        raise LeakageError(
            f"Validation period (up to {val_max}) overlaps test period "
            f"(starting {test_min}) — splits must be strictly chronological."
        )


def assert_feature_not_from_future(
    feature_timestamp: pd.Timestamp, prediction_timestamp: pd.Timestamp
) -> None:
    """Verify a single feature's source timestamp is at or before prediction time.

    For a prediction made using information as of `prediction_timestamp`,
    every feature must be observable at or before that instant — never after.
    """
    if feature_timestamp > prediction_timestamp:
        raise LeakageError(
            f"Feature timestamp {feature_timestamp} is after prediction "
            f"timestamp {prediction_timestamp} — this feature would not "
            "have been observable at prediction time."
        )


def assert_horizon_target_is_future(
    feature_cutoff: pd.Timestamp, target_timestamp: pd.Timestamp, horizon: pd.Timedelta
) -> None:
    """Verify a training target sits exactly `horizon` after the feature cutoff.

    Guards against accidentally building (features, target) pairs where the
    target is not actually `horizon` in the future of the features it's
    paired with (e.g. an off-by-one in a lag/shift operation).
    """
    expected = feature_cutoff + horizon
    if target_timestamp != expected:
        raise LeakageError(
            f"Target timestamp {target_timestamp} is not exactly {horizon} "
            f"after feature cutoff {feature_cutoff} (expected {expected})."
        )
