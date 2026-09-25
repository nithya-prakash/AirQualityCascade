import pytest
from fastapi.testclient import TestClient

from aqcascade.api.main import app
from tests.conftest import LOCAL_DATA_AVAILABLE, SKIP_REASON_NO_DATA

pytestmark = pytest.mark.skipif(not LOCAL_DATA_AVAILABLE, reason=SKIP_REASON_NO_DATA)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # triggers the lifespan startup (loads the registry once)
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["n_stations"] > 0
    assert "pm25_h1" in body["models_loaded"]


def test_predict_valid_request_returns_real_prediction(client):
    resp = client.post("/predict", json={"station_id": 21, "pollutant": "pm25", "horizon_hours": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["station_id"] == 21
    assert body["horizon_hours"] == 1
    assert body["pollutant"] == "pm25"
    assert isinstance(body["predicted_value"], float)
    assert body["predicted_value"] >= 0  # a real PM2.5 prediction, never negative
    assert body["uncertainty"] is not None
    assert body["uncertainty"]["mae"] > 0


def test_predict_defaults_to_pm25_h1_and_latest_data(client):
    resp = client.post("/predict", json={"station_id": 21})
    assert resp.status_code == 200
    body = resp.json()
    assert body["pollutant"] == "pm25"
    assert body["horizon_hours"] == 1
    assert body["as_of_timestamp"] == "2026-09-21T23:00:00"  # the dataset's actual last hour


def test_predict_unknown_station_returns_404(client):
    resp = client.post("/predict", json={"station_id": 999999999})
    assert resp.status_code == 404
    assert "999999999" in resp.json()["detail"]


def test_predict_unavailable_model_combo_returns_422(client):
    # NO2 t+2h was never trained (see README Phase 5 scope note).
    resp = client.post("/predict", json={"station_id": 21, "pollutant": "no2", "horizon_hours": 2})
    assert resp.status_code == 422


def test_predict_rejects_invalid_horizon_via_pydantic(client):
    # horizon_hours must be 1 or 2 -- this should fail validation before
    # ever reaching the model registry.
    resp = client.post("/predict", json={"station_id": 21, "horizon_hours": 5})
    assert resp.status_code == 422


def test_predict_rejects_invalid_pollutant_via_pydantic(client):
    resp = client.post("/predict", json={"station_id": 21, "pollutant": "ozone"})
    assert resp.status_code == 422


def test_predict_rejects_missing_station_id(client):
    resp = client.post("/predict", json={"pollutant": "pm25"})
    assert resp.status_code == 422


def test_model_info_valid(client):
    resp = client.get("/model-info", params={"pollutant": "pm25", "horizon_hours": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["architecture"].startswith("XGBoost")
    assert body["n_features"] > 0
    assert body["test_metrics"]["r2"] > 0  # a real, computed R^2, not a placeholder


def test_model_info_unavailable_combo_returns_404(client):
    resp = client.get("/model-info", params={"pollutant": "no2", "horizon_hours": 2})
    assert resp.status_code == 404


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "pm25_h1" in body["models"]
    assert body["models"]["pm25_h1"]["mae"] > 0
    # NO2 was never part of the Phase 8 ablation (PM2.5-only, see README),
    # so its metrics must fall back to the Phase 5 baseline results rather
    # than silently coming back null.
    assert body["models"]["no2_h1"] is not None
    assert body["models"]["no2_h1"]["mae"] > 0


def test_no2_metrics_are_the_xgboost_row_not_whichever_model_came_first(client):
    # baseline_results.csv has one row per algorithm (persistence, linear
    # regression, random forest, xgboost) per target -- the served model is
    # XGBoost, so its reported R^2 must match that specific row, not
    # persistence's (which happens to be listed first).
    resp = client.get("/metrics")
    no2_metrics = resp.json()["models"]["no2_h1"]
    assert no2_metrics["r2"] == pytest.approx(0.846, abs=0.01)  # Phase 5's real XGBoost result
