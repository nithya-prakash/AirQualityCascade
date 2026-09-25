import pandas as pd
import pytest

from aqcascade.models.feature_groups import (
    MODEL_A_COLUMNS,
    MODEL_B_COLUMNS,
    MODEL_C_COLUMNS,
    validate_against,
)
from tests.conftest import LOCAL_DATA_AVAILABLE, SKIP_REASON_NO_DATA


def test_model_tiers_are_strictly_nested():
    a, b, c = set(MODEL_A_COLUMNS), set(MODEL_B_COLUMNS), set(MODEL_C_COLUMNS)
    assert a < b < c  # each tier is a strict superset of the previous one


def test_no_duplicate_columns_within_a_tier():
    assert len(MODEL_A_COLUMNS) == len(set(MODEL_A_COLUMNS))
    assert len(MODEL_B_COLUMNS) == len(set(MODEL_B_COLUMNS))
    assert len(MODEL_C_COLUMNS) == len(set(MODEL_C_COLUMNS))


@pytest.mark.skipif(not LOCAL_DATA_AVAILABLE, reason=SKIP_REASON_NO_DATA)
def test_validate_against_real_features_parquet_columns():
    # Guards against a typo'd column name silently training Model C on
    # fewer features than intended -- validated against the actual Phase 4
    # output, not a hand-maintained copy of its schema.
    features = pd.read_parquet("data/processed/features.parquet")
    validate_against(features.columns.tolist())  # should not raise


def test_validate_against_raises_on_missing_column():
    with pytest.raises(ValueError, match="not in the data"):
        validate_against(["pm25"])  # missing almost everything
