"""Calendar/temporal features derived from the feature-cutoff timestamp T.

Deliberately derived from T (the time features are computed "as of"), not
from the target time T+horizon. hour-of-target is fully determined by
hour-of-T plus the (fixed, known) horizon, so using T avoids ever having to
reason about whether a "target-time" feature secretly leaks information —
everything here is a pure function of what's already known at T.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NORTHERN_HEMISPHERE_SEASON = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}


def add_temporal_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    ts = df[ts_col]

    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek  # Monday=0
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    df["month"] = ts.dt.month
    df["season"] = df["month"].map(NORTHERN_HEMISPHERE_SEASON)

    # Cyclical encodings so e.g. hour 23 and hour 0 are numerically adjacent.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df
