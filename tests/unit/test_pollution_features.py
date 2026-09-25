import numpy as np
import pandas as pd
import pytest

from aqcascade.features.pollution import (
    add_lag_features,
    add_rate_of_change,
    add_rolling_features,
    add_trend,
)


def _single_station_frame(values: list[float]) -> pd.DataFrame:
    hours = pd.date_range("2026-01-01", periods=len(values), freq="h")
    return pd.DataFrame({"station_id": 1, "timestamp": hours, "pm25": values})


def test_lag_1h_equals_previous_row():
    df = _single_station_frame([10.0, 12.0, 8.0, 20.0])
    out = add_lag_features(df, "pm25", [1, 2])
    assert out["pm25_lag_1h"].tolist()[1:] == [10.0, 12.0, 8.0]
    assert np.isnan(out["pm25_lag_1h"].iloc[0])  # nothing before the first row
    assert out["pm25_lag_2h"].tolist()[2:] == [10.0, 12.0]


def test_lag_never_crosses_station_boundary():
    a = _single_station_frame([1.0, 2.0, 3.0])
    b = _single_station_frame([100.0, 200.0, 300.0])
    b["station_id"] = 2
    df = pd.concat([a, b], ignore_index=True)
    out = add_lag_features(df, "pm25", [1])
    first_row_of_station_2 = out[out.station_id == 2].iloc[0]
    assert np.isnan(first_row_of_station_2["pm25_lag_1h"])  # not station 1's last value


def test_rolling_mean_matches_hand_computed_window():
    df = _single_station_frame([10.0, 20.0, 30.0, 40.0])
    out = add_rolling_features(df, "pm25", [3])
    # window ends at (and includes) the current row
    assert out["pm25_rollmean_3h"].iloc[2] == pytest.approx((10 + 20 + 30) / 3)
    assert out["pm25_rollmean_3h"].iloc[3] == pytest.approx((20 + 30 + 40) / 3)
    assert out["pm25_rollmax_3h"].iloc[3] == 40.0
    assert out["pm25_rollmin_3h"].iloc[3] == 20.0


def test_rate_of_change_is_hour_over_hour_diff():
    df = _single_station_frame([10.0, 15.0, 5.0])
    out = add_rate_of_change(df, "pm25")
    assert np.isnan(out["pm25_roc_1h"].iloc[0])
    assert out["pm25_roc_1h"].iloc[1] == 5.0
    assert out["pm25_roc_1h"].iloc[2] == -10.0


def test_trend_is_average_hourly_change_over_window():
    df = _single_station_frame([10.0, 12.0, 14.0, 16.0])
    out = add_trend(df, "pm25", window_hours=3)
    # (16 - 10) / 3 at the last row
    assert out["pm25_trend_3h"].iloc[3] == pytest.approx((16.0 - 10.0) / 3)
