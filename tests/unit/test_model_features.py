import pandas as pd

from aqcascade.models.features import build_feature_matrix, get_feature_columns


def test_get_feature_columns_excludes_identifiers_and_targets():
    df = pd.DataFrame(
        {
            "station_id": [1],
            "timestamp": [pd.Timestamp("2026-01-01")],
            "pm25": [5.0],
            "target_pm25_h1": [6.0],
        }
    )
    cols = get_feature_columns(df)
    assert cols == ["pm25"]


def test_season_dummies_are_consistent_even_when_a_split_misses_a_season():
    # Simulates a test split confined to autumn while training data spans
    # both summer and autumn -- both must end up with the same column set.
    train = pd.DataFrame({"season": ["summer", "autumn"], "pm25": [1.0, 2.0]})
    test = pd.DataFrame({"season": ["autumn"], "pm25": [3.0]})

    X_train = build_feature_matrix(train, feature_cols=["season", "pm25"])
    X_test = build_feature_matrix(test, feature_cols=["season", "pm25"])

    assert set(X_train.columns) == set(X_test.columns)
    assert "season_summer" in X_train.columns
    assert "season_summer" in X_test.columns  # present (all zeros), not silently dropped
    assert X_test["season_summer"].tolist() == [0.0]
    assert X_test["season_autumn"].tolist() == [1.0]


def test_bool_columns_become_float():
    df = pd.DataFrame({"is_weekend": [True, False], "pm25": [1.0, 2.0]})
    X = build_feature_matrix(df, feature_cols=["is_weekend", "pm25"])
    assert X["is_weekend"].tolist() == [1.0, 0.0]
    assert X["is_weekend"].dtype == float


def test_object_dtype_bool_column_is_also_converted():
    # Regularizing onto the full hourly grid (an outer join) silently turns
    # a bool column into dtype object holding {True, False, None} instead
    # of dtype bool -- must still be caught and converted, or XGBoost
    # rejects the whole column.
    df = pd.DataFrame({"pm25_is_outlier": pd.Series([True, False, None], dtype=object)})
    X = build_feature_matrix(df, feature_cols=["pm25_is_outlier"])
    assert X["pm25_is_outlier"].dtype == float
    assert X["pm25_is_outlier"].tolist()[:2] == [1.0, 0.0]
    import math

    assert math.isnan(X["pm25_is_outlier"].iloc[2])
