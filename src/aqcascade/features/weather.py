"""Weather features: the raw values already in the panel, plus a couple of
lightly-derived meteorological signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_weather_derived_features(
    df: pd.DataFrame, pressure_tendency_window_hours: int
) -> pd.DataFrame:
    df = df.copy()

    # Wind direction is circular (0deg == 360deg) — a linear feature would
    # tell the model that 1deg and 359deg are as far apart as possible,
    # which is wrong. Standard fix: sin/cos decomposition.
    radians = np.deg2rad(df["wind_direction_deg"])
    df["wind_direction_sin"] = np.sin(radians)
    df["wind_direction_cos"] = np.cos(radians)

    # Pressure tendency: falling pressure often precedes worse dispersion
    # conditions (and vice versa) — a standard, cheap meteorological signal.
    lagged_pressure = df.groupby("station_id", sort=False)["pressure_msl_hpa"].shift(
        pressure_tendency_window_hours
    )
    df[f"pressure_tendency_{pressure_tendency_window_hours}h"] = (
        df["pressure_msl_hpa"] - lagged_pressure
    )

    return df
