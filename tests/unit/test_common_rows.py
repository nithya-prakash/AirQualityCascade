import numpy as np
import pandas as pd
import pytest

from aqcascade.evaluation.common_rows import (
    KEY,
    align,
    score_models,
    select_common_rows,
    series_by_key,
)
from aqcascade.evaluation.metrics import compute_regression_metrics

T0 = pd.Timestamp("2026-09-10 00:00")


def _frame() -> pd.DataFrame:
    # station 1 has a missing current value at hour 2; station 2 has a missing h2 target at hour 3
    rows = []
    for station in (1, 2):
        for h in range(5):
            rows.append(
                {
                    "station_id": station,
                    "timestamp": T0 + pd.Timedelta(hours=h),
                    "pm25": 5.0 + h,
                    "target_pm25_h1": 6.0 + h,
                    "target_pm25_h2": 7.0 + h,
                }
            )
    df = pd.DataFrame(rows)
    df.loc[(df.station_id == 1) & (df.timestamp == T0 + pd.Timedelta(hours=2)), "pm25"] = np.nan
    df.loc[
        (df.station_id == 2) & (df.timestamp == T0 + pd.Timedelta(hours=3)), "target_pm25_h2"
    ] = np.nan
    return df


TARGETS = ["target_pm25_h1", "target_pm25_h2"]


def test_common_rows_exclude_missing_current_value_and_missing_targets() -> None:
    keys = select_common_rows(_frame(), TARGETS)
    assert len(keys) == 8  # 10 rows minus the two deliberately broken ones
    assert not ((keys.station_id == 1) & (keys.timestamp == T0 + pd.Timedelta(hours=2))).any()
    assert not ((keys.station_id == 2) & (keys.timestamp == T0 + pd.Timedelta(hours=3))).any()


def test_duplicate_keys_are_rejected() -> None:
    df = pd.concat([_frame(), _frame().iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        select_common_rows(df, TARGETS)


def test_align_restricts_a_superset_and_follows_key_order() -> None:
    df = _frame()
    keys = select_common_rows(df, TARGETS).iloc[::-1].reset_index(drop=True)  # reversed order
    superset = series_by_key(df, np.arange(len(df)))
    aligned = align(superset, keys, "m")
    expected = [superset[(r.station_id, r.timestamp)] for r in keys.itertuples()]
    assert aligned.tolist() == expected


def test_a_model_missing_common_rows_raises_instead_of_dropping_them() -> None:
    df = _frame()
    keys = select_common_rows(df, TARGETS)
    partial = series_by_key(df, np.ones(len(df))).iloc[:-3]
    with pytest.raises(KeyError, match="no prediction"):
        align(partial, keys, "partial-model")


def test_models_are_scored_on_identical_rows_with_the_shared_metric() -> None:
    df = _frame()
    keys = select_common_rows(df, TARGETS)
    truth = series_by_key(df, df["target_pm25_h1"])
    persistence = series_by_key(df.dropna(subset=["pm25"]), df.dropna(subset=["pm25"])["pm25"])
    # A model that can also score the row persistence cannot (missing current value).
    wider = series_by_key(df, df["target_pm25_h1"] + 0.5)

    scores = score_models(keys, truth, {"persistence": persistence, "wider": wider})

    assert scores["n"].tolist() == [len(keys), len(keys)]
    y = df.set_index(KEY).loc[list(map(tuple, keys.to_numpy())), "target_pm25_h1"].to_numpy()
    expected = compute_regression_metrics(y, y + 0.5)
    got = scores.set_index("model").loc["wider"]
    assert got["mae"] == pytest.approx(expected["mae"])
    assert got["rmse"] == pytest.approx(expected["rmse"])
    assert got["r2"] == pytest.approx(expected["r2"])
