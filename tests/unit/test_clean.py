import pandas as pd

from aqcascade.preprocessing.clean import (
    deduplicate,
    drop_invalid_timestamps,
    drop_out_of_schema_rows,
    flag_outliers_iqr,
)


def test_deduplicate_keeps_first_and_counts_drops():
    df = pd.DataFrame({"station_id": [1, 1, 2], "ts": [1, 1, 2], "value": [10, 99, 20]})
    out, n_dropped = deduplicate(df, ["station_id", "ts"])
    assert n_dropped == 1
    assert len(out) == 2
    assert out.iloc[0]["value"] == 10  # first occurrence kept


def test_drop_invalid_timestamps_bounds():
    df = pd.DataFrame(
        {"ts": pd.to_datetime(["2026-01-01", "2020-01-01", "2026-01-02"]), "v": [1, 2, 3]}
    )
    out, n_dropped = drop_invalid_timestamps(
        df, "ts", pd.Timestamp("2025-06-01"), pd.Timestamp("2026-06-01")
    )
    assert n_dropped == 1
    assert (out["ts"] >= pd.Timestamp("2025-06-01")).all()


def test_flag_outliers_iqr_does_not_drop_rows():
    df = pd.DataFrame({"value": [5, 6, 5, 6, 5, 6, 5, 500]})
    flags = flag_outliers_iqr(df, "value")
    assert len(flags) == len(df)
    assert flags.iloc[-1]  # the 500 is flagged
    assert not flags.iloc[0]  # normal values are not


def test_drop_out_of_schema_rows_keeps_nan():
    df = pd.DataFrame({"value": [5.0, -1.0, float("nan"), 9999.0, 10.0]})
    out, n_dropped = drop_out_of_schema_rows(df, "value", lower=0, upper=2000)
    assert n_dropped == 2  # -1.0 and 9999.0
    assert out["value"].isna().sum() == 1  # NaN preserved, not dropped
    assert len(out) == 3
