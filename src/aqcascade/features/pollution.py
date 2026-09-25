"""Lag/rolling/trend features from a pollutant's own history.

Every function here operates on a panel already regularized onto a full
hourly grid per station (see regularize.py) and groups by station_id before
shifting/rolling, so a value is never pulled across a station boundary or
across a calendar gap. All windows are trailing and inclusive of the
current row (the most recent observation at time T is legitimately known
at T) — nothing here ever looks forward.
"""

from __future__ import annotations

import pandas as pd


def add_lag_features(
    df: pd.DataFrame, value_col: str, lags_hours: list[int]
) -> pd.DataFrame:
    df = df.copy()
    grouped = df.groupby("station_id", sort=False)[value_col]
    for lag in lags_hours:
        df[f"{value_col}_lag_{lag}h"] = grouped.shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame, value_col: str, windows_hours: list[int]
) -> pd.DataFrame:
    """Trailing rolling mean/std/max/min, window ending at (and including) the current row."""
    df = df.copy()
    grouped = df.groupby("station_id", sort=False)[value_col]
    for window in windows_hours:
        rolling = grouped.rolling(window=window, min_periods=1)
        df[f"{value_col}_rollmean_{window}h"] = rolling.mean().reset_index(level=0, drop=True)
        df[f"{value_col}_rollstd_{window}h"] = rolling.std().reset_index(level=0, drop=True)
        df[f"{value_col}_rollmax_{window}h"] = rolling.max().reset_index(level=0, drop=True)
        df[f"{value_col}_rollmin_{window}h"] = rolling.min().reset_index(level=0, drop=True)
    return df


def add_rate_of_change(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Hour-over-hour change: value at T minus value at T-1h."""
    df = df.copy()
    lag1 = df.groupby("station_id", sort=False)[value_col].shift(1)
    df[f"{value_col}_roc_1h"] = df[value_col] - lag1
    return df


def add_trend(df: pd.DataFrame, value_col: str, window_hours: int) -> pd.DataFrame:
    """Average hourly change over the trailing window: (value_T - value_{T-window}) / window.

    A cheap, vectorized proxy for the local slope — avoids an expensive
    per-window linear regression while still capturing "is this station
    trending up or down right now".
    """
    df = df.copy()
    lagged = df.groupby("station_id", sort=False)[value_col].shift(window_hours)
    df[f"{value_col}_trend_{window_hours}h"] = (df[value_col] - lagged) / window_hours
    return df


def add_all_pollution_features(
    df: pd.DataFrame,
    value_col: str,
    lags_hours: list[int],
    rolling_windows_hours: list[int],
    trend_window_hours: int,
) -> pd.DataFrame:
    df = add_lag_features(df, value_col, lags_hours)
    df = add_rolling_features(df, value_col, rolling_windows_hours)
    df = add_rate_of_change(df, value_col)
    df = add_trend(df, value_col, trend_window_hours)
    return df
