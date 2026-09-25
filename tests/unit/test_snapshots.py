import numpy as np
import pandas as pd

from aqcascade.graph.snapshots import (
    build_snapshots,
    compute_feature_medians,
    get_gnn_feature_columns,
    impute_with_medians,
)


def test_get_gnn_feature_columns_excludes_neighbor_summaries_and_raw_coordinates():
    cols = [
        "pm25",
        "temperature_c",
        "latitude",
        "longitude",
        "neighbor_pm25_mean",
        "neighbor_pm25_count",
        "nearest_neighbor_distance_km",
    ]
    out = get_gnn_feature_columns(cols)
    # lat/lon excluded so Model D's feature scope exactly equals Model B's
    # (local + weather + calendar) plus the graph -- see module docstring.
    assert out == ["pm25", "temperature_c"]


def test_build_snapshots_places_values_at_correct_positions():
    df = pd.DataFrame(
        {
            "station_id": [1, 2, 1, 2],
            "timestamp": pd.to_datetime(
                ["2026-01-01 00:00", "2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 01:00"]
            ),
            "pm25": [10.0, 20.0, 11.0, 21.0],
            "target_pm25_h1": [11.0, 21.0, 12.0, 22.0],
        }
    )
    timestamps, features, targets = build_snapshots(
        df, station_order=[1, 2], feature_cols=["pm25"], target_cols=["target_pm25_h1"]
    )
    assert len(timestamps) == 2
    # hour 0: station 1 (index 0) = 10.0, station 2 (index 1) = 20.0
    assert features[0, 0, 0] == 10.0
    assert features[0, 1, 0] == 20.0
    # hour 1
    assert features[1, 0, 0] == 11.0
    assert targets[1, 1, 0] == 22.0


def test_build_snapshots_leaves_missing_station_hour_as_nan():
    # Station 2 has no row at all for the second hour.
    df = pd.DataFrame(
        {
            "station_id": [1, 2, 1],
            "timestamp": pd.to_datetime(
                ["2026-01-01 00:00", "2026-01-01 00:00", "2026-01-01 01:00"]
            ),
            "pm25": [10.0, 20.0, 11.0],
            "target_pm25_h1": [11.0, 21.0, 12.0],
        }
    )
    _, features, _ = build_snapshots(
        df, station_order=[1, 2], feature_cols=["pm25"], target_cols=["target_pm25_h1"]
    )
    assert np.isnan(features[1, 1, 0])  # station 2, hour 1 -> never observed


def test_compute_feature_medians_uses_only_train_mask():
    # 2 timestamps x 1 station x 1 feature: train=[1.0], test=[1000.0].
    feature_array = np.array([[[1.0]], [[1000.0]]], dtype="float32")
    train_mask = np.array([True, False])
    medians = compute_feature_medians(feature_array, train_mask)
    assert medians[0] == 1.0  # never sees the 1000.0 "test" value


def test_impute_with_medians_only_fills_nan():
    feature_array = np.array([[[5.0], [np.nan]]], dtype="float32")
    medians = np.array([99.0], dtype="float32")
    out = impute_with_medians(feature_array, medians)
    assert out[0, 0, 0] == 5.0  # untouched
    assert out[0, 1, 0] == 99.0  # filled
