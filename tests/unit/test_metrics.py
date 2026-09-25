import numpy as np

from aqcascade.evaluation.metrics import compute_regression_metrics


def test_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = compute_regression_metrics(y, y)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["r2"] == 1.0


def test_mae_and_rmse_hand_computed():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 16.0])
    m = compute_regression_metrics(y_true, y_pred)
    assert m["mae"] == 3.0  # (2 + 4) / 2
    assert m["rmse"] == ((2**2 + 4**2) / 2) ** 0.5


def test_mape_excludes_near_zero_actuals():
    # One near-zero actual (0.5, below the 1.0 floor) should be excluded
    # from MAPE rather than blowing up the percentage or raising.
    y_true = np.array([0.5, 10.0])
    y_pred = np.array([5.0, 11.0])  # huge relative error on the near-zero row
    m = compute_regression_metrics(y_true, y_pred)
    assert m["mape_excluded_near_zero_rows"] == 1
    assert m["mape_pct"] == 10.0  # only the |10 - 11| / 10 = 10% row counts


def test_mape_is_none_when_all_actuals_are_near_zero():
    y_true = np.array([0.0, 0.2])
    y_pred = np.array([0.1, 0.1])
    m = compute_regression_metrics(y_true, y_pred)
    assert m["mape_pct"] is None
    assert m["mape_excluded_near_zero_rows"] == 2
