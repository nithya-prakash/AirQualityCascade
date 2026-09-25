"""Reindex the panel onto a full hourly grid per station.

Why this matters for leakage: lag/rolling features are computed by *row
position* (`.shift(1)`, `.rolling(3)`), which is only correct if row N-1
really is one clock hour before row N. The raw panel only has rows for
hours a station actually reported (97.7% complete — see README), so a
naive shift on the sparse table would silently pull in the wrong hour
whenever a station has a gap. Reindexing onto every hour in the window
first (leaving genuinely absent hours as explicit NaN, never fabricated)
makes every later `.shift()`/`.rolling()` call calendar-correct.
"""

from __future__ import annotations

import pandas as pd


def regularize_hourly_grid(
    panel: pd.DataFrame,
    station_meta: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    """One row per (station_id, hour) for every station in `station_meta`,
    for every hour in [window_start, window_end], left-joined with whatever
    data the panel actually has for that hour.
    """
    hours = pd.date_range(window_start, window_end, freq="h")
    station_ids = station_meta["station_id"].unique()

    grid = pd.MultiIndex.from_product(
        [station_ids, hours], names=["station_id", "timestamp"]
    ).to_frame(index=False)

    out = grid.merge(panel, on=["station_id", "timestamp"], how="left")
    out = out.merge(
        station_meta[["station_id", "latitude", "longitude"]],
        on="station_id",
        how="left",
        suffixes=("", "_meta"),
    )
    # Coordinates come from station_meta unconditionally (they're constant
    # per station and the panel already carried them too) — prefer the
    # meta copy so newly-added grid rows aren't left with NaN coordinates.
    if "latitude_meta" in out.columns:
        out["latitude"] = out["latitude_meta"]
        out["longitude"] = out["longitude_meta"]
        out = out.drop(columns=["latitude_meta", "longitude_meta"])

    return out.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
