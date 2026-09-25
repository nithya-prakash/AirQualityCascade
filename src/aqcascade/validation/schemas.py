"""Pandera schemas for every stage of the data pipeline.

Ranges here are physical/plausibility bounds, not statistical outlier
thresholds — a schema failure means the data is impossible (e.g. a negative
concentration, a humidity above 100%), not merely unusual. Statistical
outliers (e.g. a genuine pollution spike) are flagged separately in
preprocessing/clean.py and are never rejected by these schemas.
"""

from __future__ import annotations

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

# Bounding box around Germany with a buffer, to catch gross coordinate errors.
GERMANY_LAT_RANGE = (46.0, 56.0)
GERMANY_LON_RANGE = (5.0, 16.0)

# UBA scope id confirmed against /scopes/json: 2 = "1SMW" one-hour average.
SCOPE_HOURLY = 2

# Pollutant concentrations above this are not physically plausible for
# ambient air measurements; anything higher indicates a data error, not a
# real (if extreme) pollution event. Real spikes observed in this dataset
# topped out at 362 µg/m3 for PM2.5, well under this ceiling.
MAX_PLAUSIBLE_CONCENTRATION = 2000.0

uba_measures_schema = DataFrameSchema(
    {
        "station_id": Column(int, Check.ge(1)),
        "component_id": Column(int),
        "scope_id": Column(int, Check.eq(SCOPE_HOURLY)),
        "value": Column(
            float,
            checks=[Check.ge(0), Check.le(MAX_PLAUSIBLE_CONCENTRATION)],
            nullable=True,
        ),
        "ts_start": Column("datetime64[us]"),
        "ts_end": Column("datetime64[us]"),
        "quality_index": Column(str, nullable=True),
    },
    checks=Check(
        lambda df: (df["ts_end"] - df["ts_start"]) == pd.Timedelta(hours=1),
        name="ts_end_is_exactly_one_hour_after_ts_start",
    ),
    strict=False,
    coerce=False,
)

uba_station_schema = DataFrameSchema(
    {
        "station_id": Column(int, Check.ge(1), unique=True),
        "latitude": Column(float, Check.in_range(*GERMANY_LAT_RANGE)),
        "longitude": Column(float, Check.in_range(*GERMANY_LON_RANGE)),
    },
    strict=False,
)

dwd_weather_schema = DataFrameSchema(
    {
        "dwd_station_id": Column(int, Check.ge(1)),
        "timestamp": Column("datetime64[us]"),
        "parameter": Column(
            str, Check.isin(["air_temperature", "wind", "pressure", "precipitation"])
        ),
        "temperature_c": Column(float, Check.in_range(-40, 50), nullable=True, required=False),
        "humidity_pct": Column(float, Check.in_range(0, 100), nullable=True, required=False),
        "wind_speed_ms": Column(float, Check.in_range(0, 100), nullable=True, required=False),
        "wind_direction_deg": Column(
            float, Check.in_range(0, 360), nullable=True, required=False
        ),
        "pressure_msl_hpa": Column(
            float, Check.in_range(850, 1100), nullable=True, required=False
        ),
        "precipitation_mm": Column(float, Check.ge(0), nullable=True, required=False),
    },
    strict=False,
)

panel_schema = DataFrameSchema(
    {
        "station_id": Column(int, Check.ge(1)),
        "timestamp": Column("datetime64[us]"),
        "latitude": Column(float, Check.in_range(*GERMANY_LAT_RANGE)),
        "longitude": Column(float, Check.in_range(*GERMANY_LON_RANGE)),
        "pm25": Column(
            float, checks=[Check.ge(0), Check.le(MAX_PLAUSIBLE_CONCENTRATION)], nullable=True
        ),
        "no2": Column(
            float,
            checks=[Check.ge(0), Check.le(MAX_PLAUSIBLE_CONCENTRATION)],
            nullable=True,
            required=False,
        ),
        "temperature_c": Column(float, Check.in_range(-40, 50), nullable=True, required=False),
        "humidity_pct": Column(float, Check.in_range(0, 100), nullable=True, required=False),
        "wind_speed_ms": Column(float, Check.in_range(0, 100), nullable=True, required=False),
        "wind_direction_deg": Column(
            float, Check.in_range(0, 360), nullable=True, required=False
        ),
        "pressure_msl_hpa": Column(
            float, Check.in_range(850, 1100), nullable=True, required=False
        ),
        "precipitation_mm": Column(float, Check.ge(0), nullable=True, required=False),
    },
    unique=["station_id", "timestamp"],
    strict=False,
)
