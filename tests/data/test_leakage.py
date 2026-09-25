import pandas as pd
import pytest

from aqcascade.validation.leakage import (
    LeakageError,
    assert_chronological_split,
    assert_feature_not_from_future,
    assert_horizon_target_is_future,
)


def _frame(hours: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": pd.to_datetime(hours)})


def test_chronological_split_passes_when_strictly_ordered():
    train = _frame(["2026-01-01 00:00", "2026-01-05 00:00"])
    val = _frame(["2026-01-06 00:00", "2026-01-07 00:00"])
    test = _frame(["2026-01-08 00:00", "2026-01-09 00:00"])
    assert_chronological_split(train, val, test)  # should not raise


def test_chronological_split_rejects_train_val_overlap():
    train = _frame(["2026-01-01 00:00", "2026-01-07 00:00"])
    val = _frame(["2026-01-06 00:00", "2026-01-08 00:00"])
    test = _frame(["2026-01-09 00:00"])
    with pytest.raises(LeakageError):
        assert_chronological_split(train, val, test)


def test_chronological_split_rejects_val_test_overlap():
    train = _frame(["2026-01-01 00:00"])
    val = _frame(["2026-01-02 00:00", "2026-01-05 00:00"])
    test = _frame(["2026-01-04 00:00"])
    with pytest.raises(LeakageError):
        assert_chronological_split(train, val, test)


def test_feature_from_future_is_rejected():
    with pytest.raises(LeakageError):
        assert_feature_not_from_future(
            pd.Timestamp("2026-01-01 14:35"), pd.Timestamp("2026-01-01 14:30")
        )


def test_feature_at_or_before_prediction_time_is_fine():
    assert_feature_not_from_future(
        pd.Timestamp("2026-01-01 14:00"), pd.Timestamp("2026-01-01 14:30")
    )


def test_horizon_target_mismatch_is_rejected():
    with pytest.raises(LeakageError):
        assert_horizon_target_is_future(
            pd.Timestamp("2026-01-01 14:00"),
            pd.Timestamp("2026-01-01 16:00"),
            pd.Timedelta(hours=1),
        )
