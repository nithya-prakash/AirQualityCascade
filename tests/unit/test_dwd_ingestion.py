import pandas as pd

from aqcascade.ingestion.dwd import _replace_missing_sentinel


def test_replaces_minus_999_with_na():
    df = pd.DataFrame({"temperature_c": [5.0, -999, 12.3], "other": [1, 2, 3]})
    out = _replace_missing_sentinel(df, ["temperature_c"])
    assert out["temperature_c"].isna().sum() == 1
    assert out.loc[1, "temperature_c"] is pd.NA
    # Columns not passed in are left untouched, even if they contain -999.
    assert (out["other"] == [1, 2, 3]).all()
