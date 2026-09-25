"""Weather alignment: join DWD weather onto the UBA pollution timeline.

UBA (pollution) and DWD (weather) are different physical networks, so this
is an explicit spatial join (nearest DWD station per weather parameter, per
UBA station — computed during ingestion, see configs/data_sources.yaml),
never a claim that the two sensors are co-located. Both sources are hourly,
so the temporal join is exact (same clock hour), with no interpolation.
"""

from __future__ import annotations

import pandas as pd

# parameter -> the metric columns it contributes to the aligned panel.
PARAMETER_METRIC_COLUMNS = {
    "air_temperature": ["temperature_c", "humidity_pct"],
    "wind": ["wind_speed_ms", "wind_direction_deg"],
    "pressure": ["pressure_msl_hpa"],
    "precipitation": ["precipitation_mm"],
}
ALL_METRIC_COLUMNS = [c for cols in PARAMETER_METRIC_COLUMNS.values() for c in cols]


def collapse_weather_by_dwd_station(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the long (dwd_station_id, timestamp, parameter) rows into one
    row per (dwd_station_id, timestamp) with all available metric columns.

    Each parameter's download only populates its own metric columns (the
    rest are NaN), so per group at most one row has a non-null value for any
    given metric column — max() is a safe way to combine them without
    inventing values.
    """
    present_cols = [c for c in ALL_METRIC_COLUMNS if c in weather_df.columns]
    return weather_df.groupby(["dwd_station_id", "timestamp"], as_index=False)[present_cols].max()


def build_weather_panel(weather_df: pd.DataFrame, join_df: pd.DataFrame) -> pd.DataFrame:
    """Per-UBA-station hourly weather table, joined via nearest-DWD-station per parameter."""
    weather_by_station = collapse_weather_by_dwd_station(weather_df)

    panel: pd.DataFrame | None = None
    for parameter, metric_cols in PARAMETER_METRIC_COLUMNS.items():
        cols_present = [c for c in metric_cols if c in weather_by_station.columns]
        if not cols_present:
            continue
        param_join = join_df.loc[
            join_df["parameter"] == parameter, ["uba_station_id", "dwd_station_id"]
        ]
        param_weather = weather_by_station[["dwd_station_id", "timestamp", *cols_present]]
        merged = param_join.merge(param_weather, on="dwd_station_id", how="left").drop(
            columns="dwd_station_id"
        )
        panel = (
            merged
            if panel is None
            else panel.merge(merged, on=["uba_station_id", "timestamp"], how="outer")
        )

    assert panel is not None, "No weather parameters available to build a panel"
    return panel.rename(columns={"uba_station_id": "station_id"})


def build_panel(
    pm25_measures: pd.DataFrame,
    no2_measures: pd.DataFrame,
    weather_df: pd.DataFrame,
    join_df: pd.DataFrame,
    station_meta: pd.DataFrame,
) -> pd.DataFrame:
    """The full cleaned, spatially- and temporally-aligned station-hour panel.

    Base timeline is anchored to PM2.5 reporting hours (the primary
    target) — NO2 and weather are left-joined on, so missing values are
    explicit NaNs, never fabricated.
    """
    pm25 = pm25_measures[["station_id", "ts_start", "value"]].rename(
        columns={"ts_start": "timestamp", "value": "pm25"}
    )
    no2 = no2_measures[["station_id", "ts_start", "value"]].rename(
        columns={"ts_start": "timestamp", "value": "no2"}
    )
    weather_panel = build_weather_panel(weather_df, join_df)

    panel = pm25.merge(no2, on=["station_id", "timestamp"], how="left")
    panel = panel.merge(weather_panel, on=["station_id", "timestamp"], how="left")
    panel = panel.merge(
        station_meta[["station_id", "latitude", "longitude"]], on="station_id", how="left"
    )
    return panel
