import pandas as pd

from aqcascade.preprocessing.align import build_panel, collapse_weather_by_dwd_station


def test_collapse_weather_combines_parameters_without_inventing_values():
    df = pd.DataFrame(
        [
            {
                "dwd_station_id": 1,
                "timestamp": pd.Timestamp("2026-01-01 00:00"),
                "parameter": "air_temperature",
                "temperature_c": 5.0,
                "humidity_pct": 80.0,
                "wind_speed_ms": None,
                "wind_direction_deg": None,
                "pressure_msl_hpa": None,
                "precipitation_mm": None,
            },
            {
                "dwd_station_id": 1,
                "timestamp": pd.Timestamp("2026-01-01 00:00"),
                "parameter": "wind",
                "temperature_c": None,
                "humidity_pct": None,
                "wind_speed_ms": 3.2,
                "wind_direction_deg": 180.0,
                "pressure_msl_hpa": None,
                "precipitation_mm": None,
            },
        ]
    )
    out = collapse_weather_by_dwd_station(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["temperature_c"] == 5.0
    assert row["wind_speed_ms"] == 3.2
    assert pd.isna(row["pressure_msl_hpa"])  # never measured -> stays missing, not fabricated


def test_build_panel_left_joins_no2_and_weather_onto_pm25_timeline():
    pm25 = pd.DataFrame(
        {
            "station_id": [1, 1, 2],
            "ts_start": pd.to_datetime(
                ["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 00:00"]
            ),
            "value": [10.0, 12.0, 20.0],
        }
    )
    # Station 2 never reports NO2 -> should end up NaN, not dropped or fabricated.
    no2 = pd.DataFrame(
        {
            "station_id": [1],
            "ts_start": pd.to_datetime(["2026-01-01 00:00"]),
            "value": [15.0],
        }
    )
    weather = pd.DataFrame(
        [
            {
                "dwd_station_id": 100,
                "timestamp": pd.Timestamp("2026-01-01 00:00"),
                "parameter": "air_temperature",
                "temperature_c": 4.0,
                "humidity_pct": 70.0,
            }
        ]
    )
    join_df = pd.DataFrame(
        {
            "uba_station_id": [1, 2],
            "parameter": ["air_temperature", "air_temperature"],
            "dwd_station_id": [100, 999],
        }
    )
    station_meta = pd.DataFrame(
        {"station_id": [1, 2], "latitude": [52.5, 48.1], "longitude": [13.4, 11.6]}
    )

    panel = build_panel(pm25, no2, weather, join_df, station_meta)

    assert len(panel) == 3
    row1 = panel[
        (panel.station_id == 1) & (panel.timestamp == pd.Timestamp("2026-01-01 00:00"))
    ].iloc[0]
    assert row1["no2"] == 15.0
    assert row1["temperature_c"] == 4.0

    row2 = panel[
        (panel.station_id == 1) & (panel.timestamp == pd.Timestamp("2026-01-01 01:00"))
    ].iloc[0]
    assert pd.isna(row2["no2"])  # no NO2 reading that hour -> NaN, not interpolated

    row3 = panel[panel.station_id == 2].iloc[0]
    assert pd.isna(row3["no2"])  # station 2 never reports NO2 at all
    assert pd.isna(row3["temperature_c"])  # its DWD station (999) has no data
