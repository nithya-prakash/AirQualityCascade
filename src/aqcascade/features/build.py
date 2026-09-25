"""Orchestrates the full leakage-safe feature pipeline: panel -> ML-ready table."""

from __future__ import annotations

import pandas as pd

from aqcascade.features.calendar import add_temporal_features
from aqcascade.features.pollution import add_all_pollution_features
from aqcascade.features.regularize import regularize_hourly_grid
from aqcascade.features.spatial import add_spatial_features, compute_neighbor_graph
from aqcascade.features.targets import add_targets
from aqcascade.features.weather import add_weather_derived_features


def build_feature_table(
    panel: pd.DataFrame, station_meta: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (feature_table, neighbor_graph)."""
    window_start, window_end = panel["timestamp"].min(), panel["timestamp"].max()
    df = regularize_hourly_grid(panel, station_meta, window_start, window_end)

    pollution_cfg = config["pollution_features"]
    df = add_all_pollution_features(
        df, "pm25",
        pollution_cfg["lags_hours"],
        pollution_cfg["rolling_windows_hours"],
        pollution_cfg["trend_window_hours"],
    )
    df = add_all_pollution_features(
        df, "no2",
        pollution_cfg["lags_hours"],
        pollution_cfg["rolling_windows_hours"],
        pollution_cfg["trend_window_hours"],
    )

    df = add_temporal_features(df, ts_col="timestamp")

    weather_cfg = config["weather_features"]
    df = add_weather_derived_features(df, weather_cfg["pressure_tendency_window_hours"])

    spatial_cfg = config["spatial_features"]
    neighbor_graph = compute_neighbor_graph(
        station_meta, spatial_cfg["k_neighbors"], spatial_cfg["max_neighbor_distance_km"]
    )
    df = add_spatial_features(df, neighbor_graph, "pm25")

    df = add_targets(df, "pm25", config["targets"]["pm25"]["horizons_hours"])
    df = add_targets(df, "no2", config["targets"]["no2"]["horizons_hours"])

    return df, neighbor_graph
