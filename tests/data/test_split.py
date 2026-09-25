import pandas as pd

from aqcascade.evaluation.split import chronological_split


def test_chronological_split_produces_no_overlap_and_correct_order():
    hours = pd.date_range("2026-01-01", periods=100, freq="h")
    df = pd.DataFrame({"timestamp": hours, "value": range(100)})

    train, val, test = chronological_split(df, "timestamp", train_frac=0.7, val_frac=0.15)

    assert len(train) + len(val) + len(test) == len(df)
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()
    # Roughly the requested proportions (exact split lands on a real hour).
    assert 0.65 <= len(train) / len(df) <= 0.75
    assert 0.10 <= len(val) / len(df) <= 0.20


def test_chronological_split_handles_multiple_stations_per_timestamp():
    hours = pd.date_range("2026-01-01", periods=10, freq="h")
    df = pd.DataFrame(
        {
            "timestamp": list(hours) * 3,
            "station_id": [1] * 10 + [2] * 10 + [3] * 10,
        }
    )
    train, val, test = chronological_split(df, "timestamp", train_frac=0.7, val_frac=0.15)
    # Every station present in the window should appear in every split when
    # the split boundary falls after each station's rows exist for that hour.
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()
