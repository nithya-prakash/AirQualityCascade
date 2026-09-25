"""Phase 8: error analysis for the PM2.5 t+1h test-period predictions.

Refits Model C (XGBoost, local + weather + spatial -- the strongest flat
model from the Phase 8 ablation) to get row-level predictions with
station_id/timestamp attached, which none of the training scripts save by
default (they only need aggregate metrics). Breaks errors down by station
and by time period to find where the model actually struggles, rather than
reporting only a single aggregate number.

Usage:
    python scripts/error_analysis.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from aqcascade.evaluation.spike_metrics import (  # noqa: E402
    compute_spike_classification_metrics,
    compute_spike_threshold,
)
from aqcascade.evaluation.split import chronological_split  # noqa: E402
from aqcascade.models.baselines import make_xgboost  # noqa: E402
from aqcascade.models.feature_groups import MODEL_C_COLUMNS  # noqa: E402
from aqcascade.models.features import build_feature_matrix  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
TARGET_COL = "target_pm25_h1"


def main() -> None:
    logger.info("Loading features + station metadata...")
    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    station_meta = pd.read_parquet(PROCESSED_DIR / "station_metadata.parquet")

    train, _val, test = chronological_split(features, "timestamp", train_frac=0.7, val_frac=0.15)
    train_t = train.dropna(subset=[TARGET_COL])
    test_t = test.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    logger.info(
        "Refitting Model C (XGBoost, %d features) for row-level predictions...",
        len(MODEL_C_COLUMNS),
    )
    X_train = build_feature_matrix(train_t, MODEL_C_COLUMNS)
    X_test = build_feature_matrix(test_t, MODEL_C_COLUMNS)
    model = make_xgboost()
    model.fit(X_train, train_t[TARGET_COL])
    preds = model.predict(X_test)

    result = test_t[["station_id", "timestamp", "pm25", TARGET_COL, "pm25_is_outlier"]].copy()
    result["predicted"] = preds
    result["abs_error"] = (result[TARGET_COL] - result["predicted"]).abs()
    result["hour"] = result["timestamp"].dt.hour
    result["date"] = result["timestamp"].dt.date

    # ---- Per-station breakdown ----
    by_station = (
        result.groupby("station_id")
        .agg(
            mae=("abs_error", "mean"),
            n=("abs_error", "size"),
            mean_actual=(TARGET_COL, "mean"),
            outlier_rate=("pm25_is_outlier", "mean"),
        )
        .reset_index()
    )
    station_cols = [
        "station_id",
        "station_name",
        "station_city",
        "station_type_name",
        "station_setting_name",
        "latitude",
        "longitude",
    ]
    by_station = by_station.merge(station_meta[station_cols], on="station_id", how="left")
    by_station = by_station.sort_values("mae", ascending=False).reset_index(drop=True)
    by_station.to_csv(PROCESSED_DIR / "error_analysis_by_station.csv", index=False)

    # ---- Per-hour-of-day breakdown ----
    by_hour = result.groupby("hour").agg(mae=("abs_error", "mean"), n=("abs_error", "size"))
    by_hour = by_hour.reset_index().sort_values("mae", ascending=False)
    by_hour.to_csv(PROCESSED_DIR / "error_analysis_by_hour.csv", index=False)

    # ---- Per-day breakdown ----
    by_day = result.groupby("date").agg(
        mae=("abs_error", "mean"), n=("abs_error", "size"), max_actual=(TARGET_COL, "max")
    )
    by_day = by_day.reset_index().sort_values("mae", ascending=False)
    by_day.to_csv(PROCESSED_DIR / "error_analysis_by_day.csv", index=False)

    # ---- Does error correlate with station "type" (traffic vs background etc.)? ----
    by_type = (
        result.merge(station_meta[["station_id", "station_type_name"]], on="station_id")
        .groupby("station_type_name")
        .agg(mae=("abs_error", "mean"), n=("abs_error", "size"))
        .reset_index()
        .sort_values("mae", ascending=False)
    )

    # ---- Does per-station error correlate with how often that station has outlier spikes? ----
    outlier_corr = float(by_station[["mae", "outlier_rate"]].corr().iloc[0, 1])

    # ---- Optional: pollution-spike classification (early-warning framing) ----
    spike_threshold = compute_spike_threshold(train_t[TARGET_COL].to_numpy(), percentile=90.0)
    spike_metrics = compute_spike_classification_metrics(
        result[TARGET_COL].to_numpy(), result["predicted"].to_numpy(), spike_threshold
    )

    summary = {
        "target": TARGET_COL,
        "model": f"Model C (XGBoost, {len(MODEL_C_COLUMNS)} features)",
        "test_rows": int(len(result)),
        "test_stations": int(result["station_id"].nunique()),
        "overall_mae": float(result["abs_error"].mean()),
        "worst_5_stations": by_station.head(5)[
            ["station_id", "station_name", "station_type_name", "mae", "n"]
        ].to_dict(orient="records"),
        "best_5_stations": by_station.tail(5)[
            ["station_id", "station_name", "station_type_name", "mae", "n"]
        ].to_dict(orient="records"),
        "worst_5_days": by_day.head(5)[["date", "mae", "n", "max_actual"]]
        .astype({"date": str})
        .to_dict(orient="records"),
        "error_by_station_type": by_type.to_dict(orient="records"),
        "error_by_hour_of_day": by_hour.sort_values("hour").to_dict(orient="records"),
        "correlation_station_mae_vs_outlier_rate": outlier_corr,
        "spike_classification": spike_metrics,
    }
    (PROCESSED_DIR / "error_analysis_summary.json").write_text(json.dumps(summary, indent=2))

    logger.info("Overall test MAE: %.3f", summary["overall_mae"])
    logger.info(
        "Worst 5 stations:\n%s",
        by_station.head(5)[
            ["station_id", "station_name", "station_type_name", "mae", "n"]
        ].to_string(index=False),
    )
    logger.info(
        "Best 5 stations:\n%s",
        by_station.tail(5)[
            ["station_id", "station_name", "station_type_name", "mae", "n"]
        ].to_string(index=False),
    )
    logger.info("Error by station type:\n%s", by_type.to_string(index=False))
    logger.info(
        "Worst 5 days:\n%s",
        by_day.head(5)[["date", "mae", "n", "max_actual"]].to_string(index=False),
    )
    logger.info("MAE-vs-outlier-rate correlation across stations: %.3f", outlier_corr)
    logger.info(
        "Spike classification (threshold=%.1f ug/m3, train 90th pct): %s",
        spike_threshold,
        spike_metrics,
    )
    logger.info("Wrote error_analysis_{by_station,by_hour,by_day}.csv and _summary.json")


if __name__ == "__main__":
    main()
