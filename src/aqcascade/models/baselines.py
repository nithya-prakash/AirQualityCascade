"""Baseline forecasting models, from simplest to most capable.

All four share the same (X, y) interface so scripts/train_baselines.py can
loop over them uniformly. Linear Regression and Random Forest cannot accept
NaN, so they're wrapped with a median imputer *fit on the training split
only* (fitting on val/test statistics would itself be a form of leakage —
the model would be tuned using information about data it's later "tested"
on). XGBoost handles NaN natively, so it sees the real missingness pattern
instead of an imputed one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


class Model(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Model: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class PersistenceModel:
    """Predicts "no change": the target equals the most recent observed value.

    The standard baseline for short-horizon forecasting — any model that
    can't beat this isn't adding value. Needs no fitting.
    """

    current_value_col: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PersistenceModel:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.current_value_col].to_numpy()


def make_linear_regression() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("model", LinearRegression()),
        ]
    )


def make_random_forest(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=150,
                    max_depth=14,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_xgboost(random_state: int = 42) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
    )
