"""Spatial features: what do this station's k nearest neighbors show right now?

This directly operationalizes the project's central research question (does
neighboring-station context help?) as engineered features. Neighbor values
are read at the *same* timestamp T as the target station's own features —
never a leakage risk, since both are "as of T", exactly like the weather
join. The k-nearest-neighbor graph built here (`compute_neighbor_graph`) is
reused as-is for the GNN's adjacency structure in Phase 7.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aqcascade.common.geo import haversine_km


def compute_neighbor_graph(
    station_meta: pd.DataFrame, k: int, max_distance_km: float
) -> pd.DataFrame:
    """k nearest same-pollutant neighbors for every station, within max_distance_km.

    Returns columns: station_id, neighbor_station_id, distance_km, rank
    (rank 1 = nearest). A station may end up with fewer than k neighbors if
    not enough others fall within max_distance_km — that's reported, not
    padded with fabricated neighbors.
    """
    ids = station_meta["station_id"].to_numpy()
    lats = station_meta["latitude"].to_numpy()
    lons = station_meta["longitude"].to_numpy()
    n = len(ids)

    rows = []
    for i in range(n):
        dists = np.array([haversine_km(lats[i], lons[i], lats[j], lons[j]) for j in range(n)])
        dists[i] = np.inf  # exclude self
        order = np.argsort(dists)[:k]
        for rank, j in enumerate(order, start=1):
            if dists[j] <= max_distance_km:
                rows.append(
                    {
                        "station_id": int(ids[i]),
                        "neighbor_station_id": int(ids[j]),
                        "distance_km": float(dists[j]),
                        "rank": rank,
                    }
                )
    return pd.DataFrame(rows)


def add_spatial_features(
    df: pd.DataFrame, neighbor_graph: pd.DataFrame, value_col: str
) -> pd.DataFrame:
    """For each (station, timestamp), summarize the k nearest neighbors' `value_col`."""
    pivot = df.pivot(index="timestamp", columns="station_id", values=value_col)

    nearest_distance = (
        neighbor_graph.sort_values("rank").groupby("station_id")["distance_km"].first()
    )
    neighbors_by_station = neighbor_graph.groupby("station_id")["neighbor_station_id"].apply(list)

    frames = []
    for station_id, neighbor_ids in neighbors_by_station.items():
        present = [nid for nid in neighbor_ids if nid in pivot.columns]
        if not present:
            continue
        sub = pivot[present]
        mean_series = sub.mean(axis=1, skipna=True)
        count_series = sub.notna().sum(axis=1)
        frames.append(
            pd.DataFrame(
                {
                    "station_id": station_id,
                    "timestamp": pivot.index,
                    f"neighbor_{value_col}_mean": mean_series.to_numpy(),
                    f"neighbor_{value_col}_count": count_series.to_numpy(),
                }
            )
        )

    neighbor_features = pd.concat(frames, ignore_index=True)
    # Hour-over-hour change in the neighbor average — same "trend" idea as
    # the station's own rate-of-change feature, computed on neighbor means.
    neighbor_features = neighbor_features.sort_values(["station_id", "timestamp"])
    lag1 = neighbor_features.groupby("station_id", sort=False)[f"neighbor_{value_col}_mean"].shift(
        1
    )
    neighbor_features[f"neighbor_{value_col}_roc_1h"] = (
        neighbor_features[f"neighbor_{value_col}_mean"] - lag1
    )

    neighbor_features["nearest_neighbor_distance_km"] = neighbor_features["station_id"].map(
        nearest_distance
    )

    return df.merge(neighbor_features, on=["station_id", "timestamp"], how="left")
