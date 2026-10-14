"""Pickled XGBoost models are not portable across versions (a different major gave a different
prediction in this project), so the version must be pinned identically everywhere it is used."""

import re
from pathlib import Path

import joblib
import numpy as np
import pytest
import xgboost

from tests.conftest import LOCAL_DATA_AVAILABLE, ROOT, SKIP_REASON_NO_DATA


def _pin(text: str) -> str:
    match = re.search(r"""["']?xgboost==([0-9][^"'\s,#]*)""", text)
    assert match, "xgboost must be pinned with == (pickled models are version-specific)"
    return match.group(1)


def test_training_and_serving_pin_the_same_xgboost_version() -> None:
    training = _pin((ROOT / "pyproject.toml").read_text())
    serving = _pin((ROOT / "docker" / "requirements-serve.txt").read_text())
    constraints = _pin((ROOT / "constraints.txt").read_text())
    assert training == serving == constraints


def test_installed_xgboost_matches_the_pin() -> None:
    assert xgboost.__version__ == _pin((ROOT / "pyproject.toml").read_text())


@pytest.mark.skipif(not LOCAL_DATA_AVAILABLE, reason=SKIP_REASON_NO_DATA)
def test_saved_models_load_without_version_warnings_and_predict_finite_values() -> None:
    import warnings

    import pandas as pd

    from aqcascade.models.features import build_feature_matrix, get_feature_columns

    features = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet").head(500)
    x = build_feature_matrix(features, get_feature_columns(features))
    for model_path in sorted(
        Path(ROOT / "models" / "baselines").glob("target_pm25_h*__xgboost.joblib")
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = joblib.load(model_path)
            preds = model.predict(x)
        assert not [w for w in caught if "save_model" in str(w.message)], model_path.name
        assert np.isfinite(preds).all(), model_path.name
