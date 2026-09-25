"""Feature groupings for the Phase 8 central experiment (Model A/B/C/D).

Calendar features (hour/day-of-week/month/season and their cyclical
encodings) are included in every tier, A through D. They're not a distinct
information *source* the way pollution history, weather, and neighbor data
are -- they're deterministic, zero-cost, always-available context, and the
research question is about whether spatial data helps, not whether knowing
the time of day helps. Excluding them from Model A would conflate "no
calendar info" with "no spatial info," muddying the one comparison this
experiment exists to make.

Static station coordinates (latitude/longitude) are treated as spatial
information and only appear from Model C onward, alongside the
neighbor-aggregate features -- consistent with Model D's node features
(src/aqcascade/graph/snapshots.py), which also exclude raw coordinates so
its feature scope exactly equals Model B's plus the graph structure.
"""

from __future__ import annotations

_POLLUTANTS = ["pm25", "no2"]
_LAG_HOURS = [1, 2, 3, 6, 12, 24]
_ROLLING_WINDOWS = [3, 6, 24]
_ROLLING_STATS = ["rollmean", "rollstd", "rollmax", "rollmin"]

LOCAL_POLLUTION_COLUMNS = (
    list(_POLLUTANTS)
    + ["pm25_is_outlier"]
    + [f"{p}_lag_{h}h" for p in _POLLUTANTS for h in _LAG_HOURS]
    + [f"{p}_{stat}_{w}h" for p in _POLLUTANTS for stat in _ROLLING_STATS for w in _ROLLING_WINDOWS]
    + [f"{p}_roc_1h" for p in _POLLUTANTS]
    + [f"{p}_trend_6h" for p in _POLLUTANTS]
)

CALENDAR_COLUMNS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "season",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
]

WEATHER_COLUMNS = [
    "temperature_c",
    "humidity_pct",
    "wind_speed_ms",
    "wind_direction_deg",
    "wind_direction_sin",
    "wind_direction_cos",
    "pressure_msl_hpa",
    "precipitation_mm",
    "pressure_tendency_3h",
]

SPATIAL_COLUMNS = [
    "latitude",
    "longitude",
    "neighbor_pm25_mean",
    "neighbor_pm25_count",
    "neighbor_pm25_roc_1h",
    "nearest_neighbor_distance_km",
]

MODEL_A_COLUMNS = LOCAL_POLLUTION_COLUMNS + CALENDAR_COLUMNS
MODEL_B_COLUMNS = MODEL_A_COLUMNS + WEATHER_COLUMNS
MODEL_C_COLUMNS = MODEL_B_COLUMNS + SPATIAL_COLUMNS


def validate_against(available_columns: list[str]) -> None:
    """Raise if any feature-group column doesn't actually exist in the data
    -- catches a typo'd/renamed column immediately instead of silently
    training a model on fewer features than intended."""
    all_defined = set(MODEL_C_COLUMNS)
    missing = all_defined - set(available_columns)
    if missing:
        raise ValueError(f"Feature group references columns not in the data: {sorted(missing)}")
