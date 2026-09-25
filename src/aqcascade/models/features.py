"""Turns a features.parquet-shaped DataFrame into a model-ready X matrix.

Column policy: everything except identifiers (station_id, timestamp) and
target_* columns is a feature — including the raw current pm25/no2 values
(the most recent observation, i.e. "lag 0") and station lat/lon (static,
known position, not leakage).
"""

from __future__ import annotations

import pandas as pd

NON_FEATURE_COLUMNS = {"station_id", "timestamp"}
CATEGORICAL_COLUMNS = ["season"]
# Fixed category order so one-hot columns are identical across train/val/test
# even if a given split's date range happens not to touch every season
# (e.g. a test period confined to autumn) — otherwise get_dummies would
# silently produce a different column set per split.
SEASON_CATEGORIES = ["winter", "spring", "summer", "autumn"]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    target_cols = {c for c in df.columns if c.startswith("target_")}
    exclude = NON_FEATURE_COLUMNS | target_cols
    return [c for c in df.columns if c not in exclude]


def build_feature_matrix(df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """Numeric feature matrix: bool -> float, categoricals one-hot encoded."""
    if feature_cols is None:
        feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()

    # A bool column that's been through an outer join with missing rows
    # (e.g. pm25_is_outlier) silently becomes dtype "object" holding
    # {True, False, None} instead of dtype bool -- catch that case too, not
    # just literal bool dtype, or XGBoost rejects the column outright.
    def _is_boolish(col: pd.Series) -> bool:
        if col.dtype == bool:
            return True
        non_null = col.dropna()
        return col.dtype == object and len(non_null) > 0 and non_null.isin([True, False]).all()

    bool_cols = [c for c in X.columns if _is_boolish(X[c])]
    for c in bool_cols:
        X[c] = X[c].astype(float)

    if "season" in X.columns:
        X["season"] = pd.Categorical(X["season"], categories=SEASON_CATEGORIES)
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, dtype=float)

    return X
