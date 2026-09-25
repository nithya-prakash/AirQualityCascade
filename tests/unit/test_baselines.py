import pandas as pd

from aqcascade.models.baselines import PersistenceModel


def test_persistence_model_predicts_current_value_unchanged():
    X = pd.DataFrame({"pm25": [5.0, 12.0, 0.0]})
    model = PersistenceModel(current_value_col="pm25").fit(X, None)
    preds = model.predict(X)
    assert preds.tolist() == [5.0, 12.0, 0.0]


def test_persistence_model_uses_the_configured_column():
    X = pd.DataFrame({"pm25": [5.0], "no2": [30.0]})
    model = PersistenceModel(current_value_col="no2").fit(X, None)
    assert model.predict(X).tolist() == [30.0]
