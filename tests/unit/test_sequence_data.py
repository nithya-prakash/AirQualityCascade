import numpy as np
import pandas as pd
import pytest
import torch

from aqcascade.models.sequence_data import (
    PollutionSequenceDataset,
    apply_normalization,
    build_samples,
    build_station_arrays,
    compute_normalization_stats,
)


def _toy_df() -> pd.DataFrame:
    hours = pd.date_range("2026-01-01", periods=5, freq="h")
    rows = []
    for station_id in (1, 2):
        for i, h in enumerate(hours):
            rows.append(
                {
                    "station_id": station_id,
                    "timestamp": h,
                    "pm25": float(i) if not (station_id == 1 and i == 2) else np.nan,
                    "no2": 5.0,
                    "temperature_c": 10.0,
                    "humidity_pct": 50.0,
                    "wind_speed_ms": 2.0,
                    "wind_direction_sin": 0.0,
                    "wind_direction_cos": 1.0,
                    "pressure_msl_hpa": 1010.0,
                    "precipitation_mm": 0.0,
                    "hour_sin": 0.0,
                    "hour_cos": 1.0,
                    "day_of_week_sin": 0.0,
                    "day_of_week_cos": 1.0,
                    "target_pm25_h1": float(i + 1),
                }
            )
    return pd.DataFrame(rows)


def test_build_station_arrays_forward_fills_within_station_only():
    df = _toy_df()
    arrays, timestamps, channels = build_station_arrays(df)
    pm25_idx = channels.index("pm25")
    mask_idx = channels.index("pm25_was_missing")

    station1 = arrays[1]
    # row i=2 was NaN -> forward-filled from row i=1's value (1.0), flagged.
    assert station1[2, pm25_idx] == 1.0
    assert station1[2, mask_idx] == 1.0
    assert station1[1, mask_idx] == 0.0

    # station 2 never had a gap -> untouched, and station 1's gap must not
    # have leaked into station 2's array.
    station2 = arrays[2]
    assert list(station2[:, pm25_idx]) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_build_samples_requires_full_warmup_window_and_present_target():
    df = _toy_df()
    samples = build_samples(df, ["target_pm25_h1"], seq_len=3)
    # t_idx runs 0..4 per station; seq_len=3 needs t_idx >= 2 -> 3 rows/station.
    assert (samples.groupby("station_id").size() == 3).all()
    assert samples["t_idx"].min() == 2


def test_dataset_window_never_reaches_past_the_prediction_row():
    df = _toy_df()
    arrays, _, channels = build_station_arrays(df)
    samples = build_samples(df, ["target_pm25_h1"], seq_len=3)
    ds = PollutionSequenceDataset(samples, arrays, ["target_pm25_h1"], seq_len=3)

    window, target = ds[0]
    assert window.shape == (3, len(channels))
    row = samples.iloc[0]
    expected_last = arrays[int(row["station_id"])][int(row["t_idx"])]
    assert torch.allclose(window[-1], torch.from_numpy(expected_last))
    assert target.item() == row["target_pm25_h1"]


def test_fully_missing_column_falls_back_to_dataset_median_not_zero():
    # Station 1 never reports pressure at all; station 2 always does.
    # Forward-fill can't help station 1 -- must NOT fall back to a flat 0.0
    # (physically nonsensical for pressure, ~1000 hPa off distribution).
    hours = pd.date_range("2026-01-01", periods=3, freq="h")
    rows = []
    for station_id, pressures in [(1, [np.nan, np.nan, np.nan]), (2, [1000.0, 1010.0, 1020.0])]:
        for h, p in zip(hours, pressures, strict=True):
            rows.append(
                {
                    "station_id": station_id,
                    "timestamp": h,
                    "pm25": 5.0,
                    "no2": 5.0,
                    "temperature_c": 10.0,
                    "humidity_pct": 50.0,
                    "wind_speed_ms": 2.0,
                    "wind_direction_sin": 0.0,
                    "wind_direction_cos": 1.0,
                    "pressure_msl_hpa": p,
                    "precipitation_mm": 0.0,
                    "hour_sin": 0.0,
                    "hour_cos": 1.0,
                    "day_of_week_sin": 0.0,
                    "day_of_week_cos": 1.0,
                }
            )
    df = pd.DataFrame(rows)
    arrays, _, channels = build_station_arrays(df)
    pressure_idx = channels.index("pressure_msl_hpa")

    # Median of the real readings (station 2 only) is 1010.0.
    assert np.all(arrays[1][:, pressure_idx] == 1010.0)
    assert not np.any(arrays[1][:, pressure_idx] == 0.0)


def test_normalization_stats_computed_only_from_train_period():
    df = _toy_df()
    arrays, timestamps, channels = build_station_arrays(df)
    # Train ends after the 3rd hour (index 2) -> stats should reflect only
    # pm25 values [0,1,1(ffilled)] for station 1 and [0,1,2] for station 2,
    # never the later values 3.0/4.0.
    train_end = pd.Timestamp("2026-01-01 02:00")
    stats = compute_normalization_stats(arrays, timestamps, channels, ["pm25"], train_end)
    mean, _ = stats["pm25"]
    all_train_values = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 2.0])
    assert mean == pytest.approx(all_train_values.mean())


def test_apply_normalization_does_not_mutate_input_arrays():
    df = _toy_df()
    arrays, timestamps, channels = build_station_arrays(df)
    original = arrays[1].copy()
    stats = compute_normalization_stats(
        arrays, timestamps, channels, ["pm25"], pd.Timestamp("2026-01-01 04:00")
    )
    normalized = apply_normalization(arrays, channels, ["pm25"], stats)
    assert np.array_equal(arrays[1], original)  # untouched
    assert not np.array_equal(normalized[1], original)  # actually changed
