"""Cleaning for raw UBA/DWD data: duplicates, impossible values, outlier flags.

Policy (see README "Limitations" / data_sources.yaml): missing values are
never fabricated or filled in here. Rows with impossible values (caught by
the pandera schemas in aqcascade.validation.schemas) are dropped and counted.
Statistically unusual but physically possible values (e.g. a genuine
pollution spike) are flagged in an `is_outlier` column, never dropped —
feature engineering / modeling decides what to do with them.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def deduplicate(df: pd.DataFrame, subset: list[str]) -> tuple[pd.DataFrame, int]:
    """Drop duplicate rows on `subset`, keeping the first occurrence.

    Returns (deduplicated_df, n_dropped).
    """
    before = len(df)
    out = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    n_dropped = before - len(out)
    if n_dropped:
        logger.warning("Dropped %d duplicate rows on %s", n_dropped, subset)
    return out, n_dropped


def drop_invalid_timestamps(
    df: pd.DataFrame, ts_col: str, min_ts: pd.Timestamp, max_ts: pd.Timestamp
) -> tuple[pd.DataFrame, int]:
    """Drop rows whose timestamp falls outside a sane [min_ts, max_ts] bound.

    Catches parsing errors (e.g. a timestamp decades off) that would
    otherwise silently corrupt a chronological split.
    """
    before = len(df)
    mask = (df[ts_col] >= min_ts) & (df[ts_col] <= max_ts)
    out = df[mask].reset_index(drop=True)
    n_dropped = before - len(out)
    if n_dropped:
        logger.warning(
            "Dropped %d rows with %s outside [%s, %s]", n_dropped, ts_col, min_ts, max_ts
        )
    return out, n_dropped


def flag_outliers_iqr(df: pd.DataFrame, col: str, factor: float = 3.0) -> pd.Series:
    """Flag statistical outliers via Tukey's IQR rule (factor=3.0 -> "far outliers").

    This never removes data — it only labels values that are statistically
    unusual so downstream modeling/error-analysis can look at them
    explicitly. A wide factor (3.0 instead of the common 1.5) is used
    deliberately: pollution data is genuinely spiky, and 1.5x IQR would
    flag a large fraction of normal readings as "outliers".
    """
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return (df[col] < lower) | (df[col] > upper)


def drop_out_of_schema_rows(
    df: pd.DataFrame, value_col: str, lower: float, upper: float
) -> tuple[pd.DataFrame, int]:
    """Drop rows with a physically impossible value (outside [lower, upper]).

    NaN values are kept (missing is not the same as impossible) — only
    non-null values outside the plausible range are dropped.
    """
    before = len(df)
    impossible = df[value_col].notna() & ((df[value_col] < lower) | (df[value_col] > upper))
    if impossible.any():
        logger.warning(
            "Dropping %d rows with impossible %s (outside [%s, %s]): sample=%s",
            impossible.sum(), value_col, lower, upper,
            df.loc[impossible, value_col].head(5).tolist(),
        )
    out = df[~impossible].reset_index(drop=True)
    return out, before - len(out)
