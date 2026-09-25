"""Forward-looking prediction targets.

This is the one place in the feature pipeline that's allowed to look into
the future — by construction, not by accident: target(T) = value(T +
horizon). Every feature module elsewhere in this package only ever looks at
or before T. See tests/data/test_feature_leakage.py for a check that
verifies features and targets don't get mixed up.
"""

from __future__ import annotations

import pandas as pd


def add_targets(df: pd.DataFrame, value_col: str, horizons_hours: list[int]) -> pd.DataFrame:
    df = df.copy()
    grouped = df.groupby("station_id", sort=False)[value_col]
    for h in horizons_hours:
        df[f"target_{value_col}_h{h}"] = grouped.shift(-h)
    return df
