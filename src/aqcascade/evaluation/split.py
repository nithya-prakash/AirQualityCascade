"""Chronological train/validation/test splitting.

Random splitting is never used for evaluation in this project (see README
"Temporal leakage prevention") — every split is by time, and every split
self-verifies via aqcascade.validation.leakage.assert_chronological_split
before being returned, so a bug here fails loudly instead of silently
producing an optimistic, leaked evaluation.
"""

from __future__ import annotations

import pandas as pd

from aqcascade.validation.leakage import assert_chronological_split


def chronological_split(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by the boundary timestamps of the full time range, not by row
    count — with a regular hourly grid the two are nearly identical, but
    splitting on time is correct even when they're not.
    """
    unique_times = sorted(df[time_col].unique())
    n = len(unique_times)
    train_end = unique_times[int(n * train_frac) - 1]
    val_end = unique_times[int(n * (train_frac + val_frac)) - 1]

    train = df[df[time_col] <= train_end].reset_index(drop=True)
    val = df[(df[time_col] > train_end) & (df[time_col] <= val_end)].reset_index(drop=True)
    test = df[df[time_col] > val_end].reset_index(drop=True)

    assert_chronological_split(train, val, test, time_col)
    return train, val, test
