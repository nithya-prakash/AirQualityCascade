"""Loads trained models + the processed feature table once, and serves
single-station predictions from them.

Serving basis, stated plainly: this project's data pipeline (Phases 1-4) is
batch ingestion, not a live sensor feed -- there is no streaming connection
to UBA/DWD. A prediction is therefore always computed from the most recent
row this project has actually ingested and feature-engineered for a
station (or an explicitly requested historical timestamp), never a
simulated "live" reading. This is documented here and surfaced in every API
response via `as_of_timestamp`, rather than presented as real-time.

Models served: the Phase 5/8 XGBoost models trained on Model C's feature
set (local history + weather + neighbor spatial features) -- verified
byte-for-byte identical in column composition to Model C in the Phase 8
ablation (see scripts/run_ablation.py), so "the model behind this API" and
"the model whose accuracy is reported in the README" are the same
artifact, not a proxy for it. XGBoost was chosen for serving specifically
because it needs no separate imputation step at inference time (unlike the
Linear Regression / Random Forest baselines) and is the best- or
near-best-performing model on every target evaluated in Phase 8.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBRegressor

from aqcascade.models.features import build_feature_matrix, get_feature_columns

ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models" / "baselines"

# (pollutant, horizon_hours) -> (target column, joblib filename)
AVAILABLE_MODELS = {
    ("pm25", 1): ("target_pm25_h1", "target_pm25_h1__xgboost.joblib"),
    ("pm25", 2): ("target_pm25_h2", "target_pm25_h2__xgboost.joblib"),
    ("no2", 1): ("target_no2_h1", "target_no2_h1__xgboost.joblib"),
}

UNITS = {"pm25": "µg/m³", "no2": "µg/m³"}


class StationNotFoundError(Exception):
    pass


class ModelUnavailableError(Exception):
    pass


class ModelRegistry:
    """Loaded once at API startup and reused for every request -- reloading
    a 1MB+ XGBoost model and a 48MB feature table per request would be both
    slow and pointless, since none of it changes between requests in this
    batch-ingested project."""

    def __init__(self) -> None:
        self.features = pd.read_parquet(PROCESSED_DIR / "features.parquet")
        # The exact column SET here is identical to Model C's (verified in
        # Phase 8), but XGBoost validates exact column ORDER at predict
        # time, not just set membership -- so serving must reproduce
        # get_feature_columns()'s natural ordering (what scripts/
        # train_baselines.py actually trained with), not
        # feature_groups.MODEL_C_COLUMNS's curated-for-readability order.
        # Caught by a real prediction call raising a feature_names
        # mismatch, not by inspection.
        self.model_feature_columns = get_feature_columns(self.features)
        self.station_meta = pd.read_parquet(PROCESSED_DIR / "station_metadata.parquet").set_index(
            "station_id"
        )
        self.models: dict[tuple[str, int], XGBRegressor] = {}
        for key, (_target_col, filename) in AVAILABLE_MODELS.items():
            path = MODELS_DIR / filename
            if path.exists():
                self.models[key] = joblib.load(path)

        self.ablation_results = self._load_csv(PROCESSED_DIR / "ablation_results.csv")
        self.baseline_results = self._load_csv(PROCESSED_DIR / "baseline_results.csv")
        self.data_as_of = self.features["timestamp"].max()
        self.n_stations = self.features["station_id"].nunique()

        # station_id -> row index of its most recent feature snapshot, so a
        # request that omits `as_of` doesn't have to search the whole table.
        latest_idx = self.features.groupby("station_id")["timestamp"].idxmax()
        self._latest_rows = self.features.loc[latest_idx].set_index("station_id")

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame | None:
        return pd.read_csv(path) if path.exists() else None

    def station_exists(self, station_id: int) -> bool:
        return station_id in self.station_meta.index

    def get_station_name(self, station_id: int) -> str:
        return str(self.station_meta.loc[station_id, "station_name"])

    def _get_feature_row(self, station_id: int, as_of: pd.Timestamp | None) -> pd.Series:
        if as_of is None:
            if station_id not in self._latest_rows.index:
                raise StationNotFoundError(f"No data available for station_id={station_id}")
            return self._latest_rows.loc[station_id]

        match = self.features[
            (self.features["station_id"] == station_id) & (self.features["timestamp"] == as_of)
        ]
        if match.empty:
            raise StationNotFoundError(
                f"No feature row for station_id={station_id} at as_of={as_of} "
                "(data only exists for hours actually ingested -- see /health for the "
                "available date range)"
            )
        return match.iloc[0]

    def predict(
        self, station_id: int, pollutant: str, horizon_hours: int, as_of: pd.Timestamp | None = None
    ) -> dict:
        if not self.station_exists(station_id):
            raise StationNotFoundError(f"Unknown station_id={station_id}")

        key = (pollutant, horizon_hours)
        if key not in self.models:
            available = sorted(self.models.keys())
            raise ModelUnavailableError(
                f"No trained model for pollutant={pollutant!r}, horizon_hours={horizon_hours} "
                f"-- available combinations: {available}"
            )

        row = self._get_feature_row(station_id, as_of)
        row_df = pd.DataFrame([row])
        X = build_feature_matrix(row_df, self.model_feature_columns)
        model = self.models[key]
        predicted_value = float(model.predict(X)[0])

        as_of_ts = row["timestamp"]
        prediction_ts = as_of_ts + pd.Timedelta(hours=horizon_hours)

        return {
            "station_id": station_id,
            "station_name": self.get_station_name(station_id),
            "pollutant": pollutant,
            "unit": UNITS[pollutant],
            "as_of_timestamp": as_of_ts,
            "prediction_timestamp": prediction_ts,
            "horizon_hours": horizon_hours,
            "predicted_value": round(predicted_value, 2),
            "model_version": self._model_version(key),
            "uncertainty": self._test_metrics(key),
        }

    def _model_version(self, key: tuple[str, int]) -> str:
        target_col, _filename = AVAILABLE_MODELS[key]
        return f"xgboost-model-c__{target_col}"

    def _test_metrics(self, key: tuple[str, int]) -> dict | None:
        """Real, already-computed Phase 8 test-set metrics for this model --
        used both as /model-info's reported accuracy and as /predict's
        "uncertainty" field (explicitly labeled as historical residual
        spread, not a per-prediction interval, since XGBoost point
        predictions don't produce one natively -- see README Limitations).

        Tries the Phase 8 ablation results (Model C, i.e. the exact model
        served) first, falling back to the Phase 5 baseline results for
        targets the ablation never covered (NO2 was PM2.5-only, see
        README). The fallback must also filter to the "xgboost" row
        specifically -- baseline_results.csv has one row per algorithm per
        target, and taking the first match without that filter would
        silently report whichever model happened to be listed first, not
        the one actually being served.
        """
        target_col, _filename = AVAILABLE_MODELS[key]

        if self.ablation_results is not None:
            row = self.ablation_results[
                (self.ablation_results["target"] == target_col)
                & (self.ablation_results["split"] == "test")
                & (self.ablation_results["tier"] == "C")
            ]
            if not row.empty:
                return self._format_metrics(row.iloc[0])

        if self.baseline_results is not None:
            row = self.baseline_results[
                (self.baseline_results["target"] == target_col)
                & (self.baseline_results["split"] == "test")
                & (self.baseline_results["model"] == "xgboost")
            ]
            if not row.empty:
                return self._format_metrics(row.iloc[0])

        return None

    @staticmethod
    def _format_metrics(r: pd.Series) -> dict:
        return {
            "mae": round(float(r["mae"]), 3),
            "rmse": round(float(r["rmse"]), 3),
            "r2": round(float(r["r2"]), 3),
            "note": "Historical test-set metrics (2026-09-08 -> 2026-09-21), "
            "not a per-prediction confidence interval.",
        }

    def get_model_info(self, pollutant: str, horizon_hours: int) -> dict:
        key = (pollutant, horizon_hours)
        if key not in self.models:
            raise ModelUnavailableError(f"No trained model for {pollutant} h{horizon_hours}")
        target_col, _filename = AVAILABLE_MODELS[key]
        model = self.models[key]

        info = {
            "model_version": self._model_version(key),
            "pollutant": pollutant,
            "horizon_hours": horizon_hours,
            "target_column": target_col,
            "architecture": "XGBoost (Model C: local history + weather + neighbor spatial "
            "features)",
            "n_features": len(self.model_feature_columns),
            "n_estimators": int(model.n_estimators or 0),
            "max_depth": int(model.max_depth or 0),
            "test_metrics": self._test_metrics(key),
        }
        return info

    def get_health(self) -> dict:
        return {
            "status": "ok",
            "models_loaded": [f"{p}_h{h}" for p, h in sorted(self.models.keys())],
            "n_stations": int(self.n_stations),
            "data_as_of": self.data_as_of.isoformat(),
            "data_source": "batch-ingested UBA/DWD data (see configs/data_sources.yaml); "
            "not a live sensor feed",
        }

    def get_metrics(self) -> dict:
        """Every served model's real, already-computed Phase 8 test metrics
        in one place -- not live operational metrics (request counts,
        latency), since this project has no request-monitoring stack; see
        README Limitations for that distinction.
        """
        return {
            "models": {
                f"{pollutant}_h{horizon}": self._test_metrics((pollutant, horizon))
                for pollutant, horizon in sorted(self.models.keys())
            },
            "error_analysis": self._load_json(PROCESSED_DIR / "error_analysis_summary.json"),
        }

    @staticmethod
    def _load_json(path: Path) -> dict | None:
        return json.loads(path.read_text()) if path.exists() else None
