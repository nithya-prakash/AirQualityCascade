"""End-to-end leakage check on the real feature-building pipeline.

Builds a small synthetic two-station panel with a distinctive, easy-to-spot
"spike" value planted at one specific future hour, runs it through the same
build_feature_table() used in production, and asserts that:

1. No *feature* column (own-station or neighbor) at or before the spike's
   hour ever contains the spike value — for either station, since the
   spike could leak through a station's own history OR through its
   neighbor's.
2. The *target* columns DO see the spike at the right offset — a positive
   control, proving the leakage-freedom above isn't trivially true just
   because nothing in the test ever looks anywhere.
"""

from __future__ import annotations

import pandas as pd

from aqcascade.features.build import build_feature_table

SPIKE_VALUE = 9999.0

CONFIG = {
    "targets": {"pm25": {"horizons_hours": [1, 2]}, "no2": {"horizons_hours": [1]}},
    "pollution_features": {
        "lags_hours": [1, 2, 3],
        "rolling_windows_hours": [3],
        "trend_window_hours": 3,
    },
    "spatial_features": {"k_neighbors": 2, "max_neighbor_distance_km": 100},
    "weather_features": {"pressure_tendency_window_hours": 3},
}


def _build_synthetic_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    hours = pd.date_range("2026-01-01 00:00", periods=48, freq="h")
    spike_hour = hours[30]

    station_meta = pd.DataFrame(
        {"station_id": [1, 2], "latitude": [52.50, 52.51], "longitude": [13.40, 13.40]}
    )

    rows = []
    for station_id in (1, 2):
        for h in hours:
            value = SPIKE_VALUE if (station_id == 2 and h == spike_hour) else 10.0
            rows.append(
                {
                    "station_id": station_id,
                    "timestamp": h,
                    "pm25": value,
                    "no2": 5.0,
                    "temperature_c": 15.0,
                    "humidity_pct": 60.0,
                    "wind_speed_ms": 3.0,
                    "wind_direction_deg": 180.0,
                    "pressure_msl_hpa": 1013.0,
                    "precipitation_mm": 0.0,
                    "latitude": station_meta.set_index("station_id").loc[station_id, "latitude"],
                    "longitude": station_meta.set_index("station_id").loc[station_id, "longitude"],
                }
            )
    panel = pd.DataFrame(rows)
    return panel, station_meta, spike_hour


def test_no_feature_column_ever_contains_the_future_spike_before_it_happens():
    """At the spike's own hour T, features computed "as of T" are allowed to
    see it (e.g. a rolling max that includes the current row) -- that's the
    present, not the future. The actual leakage boundary is STRICTLY before
    T: nothing computed for an earlier row may ever see it.
    """
    panel, station_meta, spike_hour = _build_synthetic_panel()

    features, _ = build_feature_table(panel, station_meta, CONFIG)

    non_feature_cols = {"station_id", "timestamp", "latitude", "longitude", "pm25", "no2"}
    target_cols = {c for c in features.columns if c.startswith("target_")}
    feature_cols = [
        c for c in features.columns if c not in non_feature_cols and c not in target_cols
    ]

    strictly_before_spike = features[features["timestamp"] < spike_hour]
    for col in feature_cols:
        assert not (strictly_before_spike[col] == SPIKE_VALUE).any(), (
            f"Feature '{col}' contains the future spike value before it occurred "
            "-- this is a temporal leakage bug."
        )


def test_target_columns_do_see_the_spike_at_the_right_offset():
    """Positive control: proves the pipeline's shift direction is correct,
    so the leakage test above isn't vacuously true."""
    panel, station_meta, spike_hour = _build_synthetic_panel()

    features, _ = build_feature_table(panel, station_meta, CONFIG)

    one_hour_before = features[
        (features.station_id == 2) & (features.timestamp == spike_hour - pd.Timedelta(hours=1))
    ].iloc[0]
    assert one_hour_before["target_pm25_h1"] == SPIKE_VALUE

    two_hours_before = features[
        (features.station_id == 2) & (features.timestamp == spike_hour - pd.Timedelta(hours=2))
    ].iloc[0]
    assert two_hours_before["target_pm25_h2"] == SPIKE_VALUE
