"""Scoring several models on one explicit, shared set of test rows.

The per-model training scripts each drop different rows before scoring: the baselines need the
current PM2.5 value (persistence copies it), the ablation tiers and the GNN only need the target,
and the Transformer needs a full 24 h window and both targets but not the current value. Their
reported metrics therefore describe different rows and cannot be ranked against each other.

This module defines the common set once and makes the alignment explicit: a model that cannot
produce a prediction for every common row is an error, never a silent drop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aqcascade.evaluation.metrics import compute_regression_metrics

KEY = ["station_id", "timestamp"]


def select_common_rows(
    test: pd.DataFrame, target_cols: list[str], current_col: str = "pm25"
) -> pd.DataFrame:
    """Keys of the test rows every compared model can be scored on.

    A row is eligible when all targets and the current value are present: persistence needs the
    current value, and every model is scored against the same non-missing targets.
    """
    eligible = test.dropna(subset=[*target_cols, current_col])
    keys = eligible[KEY].reset_index(drop=True)
    if keys.duplicated().any():
        raise ValueError("duplicate (station_id, timestamp) keys in the test split")
    return keys


def _index(keys: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(keys[KEY])


def align(values: pd.Series, keys: pd.DataFrame, what: str) -> np.ndarray:
    """Values for exactly `keys`, in `keys` order; raises if any key has no value."""
    if values.index.has_duplicates:
        raise ValueError(f"{what}: duplicate keys in the predictions")
    aligned = values.reindex(_index(keys))
    missing = int(aligned.isna().sum())
    if missing:
        raise KeyError(f"{what}: no prediction for {missing} of {len(keys)} common rows")
    return aligned.to_numpy(dtype=float)


def score_models(
    keys: pd.DataFrame, truth: pd.Series, predictions: dict[str, pd.Series]
) -> pd.DataFrame:
    """One row per model: n, mae, rmse, r2, all computed on identical rows and targets.

    `truth` and every prediction series are indexed by (station_id, timestamp); the same
    `compute_regression_metrics` is used for every model.
    """
    y = align(truth, keys, "truth")
    rows = []
    for name, series in predictions.items():
        metrics = compute_regression_metrics(y, align(series, keys, name))
        rows.append(
            {"model": name, "n": len(keys), **{k: metrics[k] for k in ("mae", "rmse", "r2")}}
        )
    return pd.DataFrame(rows)


def series_by_key(df: pd.DataFrame, values: np.ndarray | pd.Series) -> pd.Series:
    """A prediction/target series indexed by (station_id, timestamp)."""
    return pd.Series(np.asarray(values, dtype=float), index=_index(df))
