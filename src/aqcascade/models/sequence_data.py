"""Sequence windows for the Temporal Transformer.

Unlike the Phase 5 baselines (hand-engineered lag/rolling features flattened
into one row per prediction), the Transformer consumes a raw sequence of the
last `seq_len` hourly steps per station and learns temporal structure
itself — "a sequence of historical measurements and weather features", not
pre-summarized statistics of them.

Leakage-safety here is structural, not a runtime check: a window for
prediction time T is built by slicing a station's chronologically-sorted
array at positions [T-seq_len+1, ..., T] — it is physically impossible for
that slice to contain a timestamp after T, since later timestamps simply
don't exist yet at that array position. The pandas-native lag/rolling
leakage risk from Phase 4 (row order silently not matching calendar order)
still applies, so windows are built from the *same* hourly-regularized grid
`features.parquet` already validated for that in Phase 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Raw sequence channels: current pollutant values + weather + calendar.
# Deliberately excludes the Phase 4 lag/rolling/spatial features -- the
# whole point of a sequence model is to learn temporal structure from the
# raw series itself, not consume a pre-digested summary of it. Neighbor
# (spatial) features are also excluded here; that's the GNN's job (Phase 7).
SEQUENCE_FEATURE_COLUMNS = [
    "pm25",
    "no2",
    "temperature_c",
    "humidity_pct",
    "wind_speed_ms",
    "wind_direction_sin",
    "wind_direction_cos",
    "pressure_msl_hpa",
    "precipitation_mm",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]
# Columns with real, non-trivial missingness (see Phase 3 data-quality
# report) get an explicit "was this imputed" indicator channel, so the
# model can learn to trust a carried-forward value less than a fresh one.
MASK_SOURCE_COLUMNS = ["pm25", "precipitation_mm"]


def build_station_arrays(
    df: pd.DataFrame,
    feature_cols: list[str] = SEQUENCE_FEATURE_COLUMNS,
    mask_source_cols: list[str] = MASK_SOURCE_COLUMNS,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[str]]:
    """Per-station (n_hours, n_channels) float32 arrays, chronologically ordered.

    Missing values are forward-filled *within each station* (never across
    station boundaries, never using a future value) and flagged with a
    companion "_was_missing" channel. Forward-fill can't help a station
    that has literally zero real readings for a column -- and that's not
    rare: e.g. 17/315 stations never got a successful pressure join, 46/315
    never got precipitation (see the Phase 2 DWD-download-gap note in
    configs/data_sources.yaml). Filling those with a flat 0.0 was tried
    first and is wrong in a way that matters: 0 hPa is not a "neutral"
    pressure reading, it's ~1000 hPa off distribution, and would have
    corrupted the very normalization stats meant to make training well
    behaved. Falling back to each column's own dataset-wide median instead
    keeps a fully-missing station's series at a physically plausible value
    -- still flagged as missing via the mask channel, still not a claim
    that the median IS the true reading.
    """
    df = df.sort_values(["station_id", "timestamp"]).copy()

    channel_names = list(feature_cols)
    for col in mask_source_cols:
        mask_name = f"{col}_was_missing"
        df[mask_name] = df[col].isna().astype("float32")
        channel_names.append(mask_name)

    column_medians = df[feature_cols].median()
    df[feature_cols] = df.groupby("station_id", sort=False)[feature_cols].ffill()
    df[feature_cols] = df[feature_cols].fillna(column_medians)

    arrays: dict[int, np.ndarray] = {}
    timestamps: dict[int, np.ndarray] = {}
    for station_id, g in df.groupby("station_id", sort=False):
        arrays[int(station_id)] = g[channel_names].to_numpy(dtype="float32")
        timestamps[int(station_id)] = g["timestamp"].to_numpy()
    return arrays, timestamps, channel_names


def build_samples(df: pd.DataFrame, target_cols: list[str], seq_len: int) -> pd.DataFrame:
    """One row per valid (station, prediction time) pair: enough history for
    a full window, and every target present (never trained/evaluated on a
    fabricated target)."""
    df = df.sort_values(["station_id", "timestamp"]).copy()
    df["t_idx"] = df.groupby("station_id", sort=False).cumcount()

    valid = df["t_idx"] >= (seq_len - 1)
    for c in target_cols:
        valid &= df[c].notna()

    return df.loc[valid, ["station_id", "timestamp", "t_idx", *target_cols]].reset_index(drop=True)


def compute_normalization_stats(
    arrays: dict[int, np.ndarray],
    timestamps: dict[int, np.ndarray],
    channel_names: list[str],
    normalize_cols: list[str],
    train_end: pd.Timestamp,
) -> dict[str, tuple[float, float]]:
    """Per-channel (mean, std) computed ONLY from rows at-or-before
    `train_end` -- fitting normalization on val/test data would itself leak
    information about the data the model is later evaluated on, the same
    principle as the baselines' train-only imputer (Phase 5)."""
    col_idx = {c: i for i, c in enumerate(channel_names)}
    collected: dict[str, list[np.ndarray]] = {c: [] for c in normalize_cols}

    for station_id, arr in arrays.items():
        mask = timestamps[station_id] <= np.datetime64(train_end)
        if not mask.any():
            continue
        for c in normalize_cols:
            collected[c].append(arr[mask, col_idx[c]])

    stats = {}
    for c in normalize_cols:
        values = np.concatenate(collected[c]) if collected[c] else np.array([0.0])
        mean, std = float(values.mean()), float(values.std())
        stats[c] = (mean, std if std > 1e-6 else 1.0)
    return stats


def apply_normalization(
    arrays: dict[int, np.ndarray],
    channel_names: list[str],
    normalize_cols: list[str],
    stats: dict[str, tuple[float, float]],
) -> dict[int, np.ndarray]:
    """Returns NEW normalized arrays; never mutates the input (so the same
    raw arrays can be reused, e.g. to recompute stats for a different
    split)."""
    col_idx = {c: i for i, c in enumerate(channel_names)}
    out = {}
    for station_id, arr in arrays.items():
        arr = arr.copy()
        for c in normalize_cols:
            mean, std = stats[c]
            arr[:, col_idx[c]] = (arr[:, col_idx[c]] - mean) / std
        out[station_id] = arr
    return out


class PollutionSequenceDataset(Dataset):
    def __init__(
        self,
        samples: pd.DataFrame,
        station_arrays: dict[int, np.ndarray],
        target_cols: list[str],
        seq_len: int,
    ):
        self.station_ids = samples["station_id"].to_numpy()
        self.t_idxs = samples["t_idx"].to_numpy()
        self.targets = samples[target_cols].to_numpy(dtype="float32")
        self.station_arrays = station_arrays
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.station_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        station_id = int(self.station_ids[idx])
        t = int(self.t_idxs[idx])
        window = self.station_arrays[station_id][t - self.seq_len + 1 : t + 1].copy()
        return torch.from_numpy(window), torch.from_numpy(self.targets[idx].copy())
