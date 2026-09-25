"""Turns features.parquet into dense (n_timestamps, n_stations, n_features)
tensors for the GNN -- one full graph snapshot per hour, since the k-NN
graph itself is static (computed once from station coordinates) but node
features change every hour.

Node feature policy: deliberately EXCLUDES the `neighbor_*` /
`nearest_neighbor_distance_km` columns from Phase 4, AND raw
`latitude`/`longitude`. The neighbor_* columns are hand-aggregated
summaries of a station's neighbors (Model C's approach) -- handing them to
the GNN as input would let it "cheat" by reading a pre-computed answer
instead of learning spatial aggregation itself via message passing, which
is the entire point of Model D. Raw coordinates are excluded for the same
reason one level up: they're themselves a form of spatial information, and
the Phase 8 ablation needs Model D's feature scope to exactly equal Model
B's (local pollution history + weather + calendar) plus the graph -- not
"Model B plus a bit of extra spatial information the other models in the
ablation don't get". Lag/rolling/weather/calendar features (all local to
the station itself, non-positional) are fair game.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EXCLUDED_SPATIAL_COLUMNS = {"latitude", "longitude", "nearest_neighbor_distance_km"}


def get_gnn_feature_columns(all_feature_columns: list[str]) -> list[str]:
    return [
        c
        for c in all_feature_columns
        if not c.startswith("neighbor_") and c not in EXCLUDED_SPATIAL_COLUMNS
    ]


def build_snapshots(
    df: pd.DataFrame,
    station_order: list[int],
    feature_cols: list[str],
    target_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (timestamps, feature_array, target_array):
    feature_array/target_array have shape (n_timestamps, n_stations, n_cols).
    Missing (station, hour) combinations (shouldn't occur given the Phase 4
    regularized grid, but not assumed) are left as NaN, never fabricated.
    """
    station_idx = {sid: i for i, sid in enumerate(station_order)}
    df = df[df["station_id"].isin(station_idx)].copy()
    df["station_pos"] = df["station_id"].map(station_idx)

    timestamps = np.sort(df["timestamp"].unique())
    ts_pos = {t: i for i, t in enumerate(timestamps)}
    t_pos = df["timestamp"].map(ts_pos).to_numpy()
    node_pos = df["station_pos"].to_numpy()

    n_t, n_nodes = len(timestamps), len(station_order)
    feature_array = np.full((n_t, n_nodes, len(feature_cols)), np.nan, dtype="float32")
    target_array = np.full((n_t, n_nodes, len(target_cols)), np.nan, dtype="float32")

    feature_array[t_pos, node_pos, :] = df[feature_cols].to_numpy(dtype="float32")
    target_array[t_pos, node_pos, :] = df[target_cols].to_numpy(dtype="float32")

    return timestamps, feature_array, target_array


def compute_feature_medians(feature_array: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """Per-feature median computed only from train-period timestamps -- the
    same train-only discipline used for the Phase 5 baseline imputer and
    the Phase 6 Transformer's normalization stats."""
    train_values = feature_array[train_mask]  # (n_train_t, n_stations, n_features)
    flat = train_values.reshape(-1, train_values.shape[-1])
    medians = np.nanmedian(flat, axis=0)
    return np.nan_to_num(medians, nan=0.0)


def impute_with_medians(feature_array: np.ndarray, medians: np.ndarray) -> np.ndarray:
    out = feature_array.copy()
    nan_mask = np.isnan(out)
    broadcast_medians = np.broadcast_to(medians, out.shape)
    out[nan_mask] = broadcast_medians[nan_mask]
    return out
