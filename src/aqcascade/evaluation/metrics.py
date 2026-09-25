"""Regression metrics shared by every model comparison in this project."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# MAPE blows up (or is undefined) as the true value approaches zero, and
# PM2.5/NO2 readings legitimately hit 0 µg/m3. Rather than silently produce
# a meaningless huge/NaN percentage, MAPE here is computed only over rows
# with |y_true| >= this floor, and the excluded count is always reported
# alongside it so the number is never presented without its caveat.
MAPE_MIN_ACTUAL = 1.0


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    mask = np.abs(y_true) >= MAPE_MIN_ACTUAL
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = None

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape_pct": mape,
        "mape_excluded_near_zero_rows": int((~mask).sum()),
        "n": int(len(y_true)),
    }
