import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from aqcascade.validation.schemas import panel_schema, uba_measures_schema


def _valid_panel_row(**overrides):
    row = {
        "station_id": 1,
        "timestamp": pd.Timestamp("2026-01-01 00:00"),
        "latitude": 52.5,
        "longitude": 13.4,
        "pm25": 8.0,
    }
    row.update(overrides)
    return row


def test_panel_schema_accepts_valid_row():
    df = pd.DataFrame([_valid_panel_row()])
    panel_schema.validate(df, lazy=True)  # should not raise


def test_panel_schema_rejects_negative_pm25():
    df = pd.DataFrame([_valid_panel_row(pm25=-5.0)])
    with pytest.raises(SchemaErrors):
        panel_schema.validate(df, lazy=True)


def test_panel_schema_rejects_coordinates_outside_germany():
    # Roughly New York's coordinates -- a plausible bug (e.g. lat/lon swapped
    # or wrong station joined), not just a typo'd number.
    df = pd.DataFrame([_valid_panel_row(latitude=40.7, longitude=-74.0)])
    with pytest.raises(SchemaErrors):
        panel_schema.validate(df, lazy=True)


def test_panel_schema_allows_missing_weather_but_not_missing_station_id():
    df = pd.DataFrame([_valid_panel_row(temperature_c=float("nan"))])
    df["temperature_c"] = df["temperature_c"].astype(float)
    panel_schema.validate(df, lazy=True)  # missing weather is fine, not fabricated

    df_no_station = pd.DataFrame([_valid_panel_row(station_id=None)])
    with pytest.raises(SchemaErrors):
        panel_schema.validate(df_no_station, lazy=True)


def test_uba_measures_schema_rejects_wrong_scope():
    df = pd.DataFrame(
        {
            "station_id": [1],
            "component_id": [9],
            "scope_id": [1],  # 1 = daily average, not the hourly scope this project uses
            "value": [8.0],
            "ts_start": pd.to_datetime(["2026-01-01 00:00"]),
            "ts_end": pd.to_datetime(["2026-01-01 01:00"]),
            "quality_index": ["1"],
        }
    )
    with pytest.raises(SchemaErrors):
        uba_measures_schema.validate(df, lazy=True)


def test_uba_measures_schema_rejects_end_not_one_hour_after_start():
    df = pd.DataFrame(
        {
            "station_id": [1],
            "component_id": [9],
            "scope_id": [2],
            "value": [8.0],
            "ts_start": pd.to_datetime(["2026-01-01 00:00"]),
            "ts_end": pd.to_datetime(["2026-01-01 03:00"]),  # should be exactly +1h
            "quality_index": ["1"],
        }
    )
    with pytest.raises(SchemaErrors):
        uba_measures_schema.validate(df, lazy=True)
