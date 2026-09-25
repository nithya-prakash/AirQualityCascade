import pandas as pd

from aqcascade.features.calendar import add_temporal_features


def test_known_timestamps():
    # 2026-01-05 is a Monday; 2026-01-10 is a Saturday.
    df = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2026-01-05 08:00", "2026-01-10 14:00"])}
    )
    out = add_temporal_features(df)
    assert out["hour"].tolist() == [8, 14]
    assert out["day_of_week"].tolist() == [0, 5]
    assert out["is_weekend"].tolist() == [False, True]
    assert out["month"].tolist() == [1, 1]
    assert out["season"].tolist() == ["winter", "winter"]


def test_hour_cyclical_encoding_wraps_around_midnight():
    # hour 0 and hour 23 are one clock-hour apart, so their sin/cos points
    # should be much closer together than hour 0 and hour 12 (12h apart) —
    # unlike the raw `hour` column, where 0 vs 23 looks like the biggest
    # possible gap.
    df = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 23:00", "2026-01-01 12:00"])}
    )
    out = add_temporal_features(df)

    def dist(i: int, j: int) -> float:
        return (
            (out["hour_sin"].iloc[i] - out["hour_sin"].iloc[j]) ** 2
            + (out["hour_cos"].iloc[i] - out["hour_cos"].iloc[j]) ** 2
        ) ** 0.5

    assert dist(0, 1) < dist(0, 2)  # midnight-to-23:00 closer than midnight-to-noon


def test_season_mapping_covers_all_months():
    df = pd.DataFrame({"timestamp": pd.date_range("2026-01-15", periods=12, freq="MS")})
    out = add_temporal_features(df)
    assert set(out["season"]) == {"winter", "spring", "summer", "autumn"}
    assert out["season"].isna().sum() == 0
