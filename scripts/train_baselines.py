"""Phase 5: baseline forecasting models, evaluated on the same chronological
train/val/test split for every (target, model) pair.

Usage:
    python scripts/train_baselines.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import joblib  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from aqcascade.evaluation.metrics import compute_regression_metrics  # noqa: E402
from aqcascade.evaluation.split import chronological_split  # noqa: E402
from aqcascade.models.baselines import (  # noqa: E402
    PersistenceModel,
    make_linear_regression,
    make_random_forest,
    make_xgboost,
)
from aqcascade.models.features import build_feature_matrix, get_feature_columns  # noqa: E402
from aqcascade.tracking.mlflow_utils import (  # noqa: E402
    configure_mlflow,
    dataset_version,
    log_split_metrics,
    log_split_periods,
)

RANDOM_SEED = 42  # matches models.baselines' hardcoded random_state=42

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models" / "baselines"

# (target column, the "current value" column persistence should copy forward)
TARGETS = [
    ("target_pm25_h1", "pm25"),
    ("target_pm25_h2", "pm25"),
    ("target_no2_h1", "no2"),
]


def make_models(current_value_col: str) -> dict:
    return {
        "persistence": PersistenceModel(current_value_col),
        "linear_regression": make_linear_regression(),
        "random_forest": make_random_forest(),
        "xgboost": make_xgboost(),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    configure_mlflow("aqcascade-baselines")

    logger.info("Loading features...")
    features_path = PROCESSED_DIR / "features.parquet"
    features = pd.read_parquet(features_path)
    feature_cols = get_feature_columns(features)
    ds_version = dataset_version(features_path)
    logger.info(
        "%d feature columns, %d rows, dataset_version=%s",
        len(feature_cols),
        len(features),
        ds_version,
    )

    train, val, test = chronological_split(features, "timestamp", train_frac=0.7, val_frac=0.15)
    logger.info(
        "Chronological split: train=%d (%s -> %s), val=%d (%s -> %s), test=%d (%s -> %s)",
        len(train),
        train.timestamp.min(),
        train.timestamp.max(),
        len(val),
        val.timestamp.min(),
        val.timestamp.max(),
        len(test),
        test.timestamp.min(),
        test.timestamp.max(),
    )

    all_results = []
    for target_col, current_value_col in TARGETS:
        logger.info("=== Target: %s ===", target_col)
        # Every model is evaluated on exactly the same rows, so the
        # persistence baseline (which needs the current value itself) is
        # held to the same standard as the others: rows also need a
        # non-null current reading, not just a non-null target.
        required_cols = [target_col, current_value_col]
        train_t = train.dropna(subset=required_cols)
        val_t = val.dropna(subset=required_cols)
        test_t = test.dropna(subset=required_cols)
        logger.info(
            "Rows with valid target+current value: train=%d val=%d test=%d",
            len(train_t),
            len(val_t),
            len(test_t),
        )

        X_train = build_feature_matrix(train_t, feature_cols)
        X_val = build_feature_matrix(val_t, feature_cols)
        X_test = build_feature_matrix(test_t, feature_cols)
        y_train, y_val, y_test = train_t[target_col], val_t[target_col], test_t[target_col]

        for model_name, model in make_models(current_value_col).items():
            with mlflow.start_run(run_name=f"{target_col}__{model_name}"):
                mlflow.log_params(
                    {
                        "model_type": model_name,
                        "target": target_col,
                        "n_features": len(feature_cols),
                        "dataset_version": ds_version,
                        "random_seed": RANDOM_SEED
                        if model_name in ("random_forest", "xgboost")
                        else "n/a",
                    }
                )
                log_split_periods(train_t, val_t, test_t)

                t0 = time.time()
                model.fit(X_train, y_train)
                fit_seconds = time.time() - t0
                mlflow.log_metric("fit_seconds", fit_seconds)
                logger.info("Fit %s / %s in %.1fs", target_col, model_name, fit_seconds)

                for split_name, X_split, y_split in [
                    ("train", X_train, y_train),
                    ("val", X_val, y_val),
                    ("test", X_test, y_test),
                ]:
                    preds = model.predict(X_split)
                    metrics = compute_regression_metrics(y_split.to_numpy(), preds)
                    log_split_metrics(split_name, metrics)
                    all_results.append(
                        {
                            "target": target_col,
                            "model": model_name,
                            "split": split_name,
                            "fit_seconds": round(fit_seconds, 2),
                            **metrics,
                        }
                    )

                if model_name != "persistence":
                    model_path = MODELS_DIR / f"{target_col}__{model_name}.joblib"
                    joblib.dump(model, model_path)
                    mlflow.log_artifact(str(model_path))

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(PROCESSED_DIR / "baseline_results.csv", index=False)

    test_summary = results_df[results_df["split"] == "test"][
        ["target", "model", "mae", "rmse", "r2", "n"]
    ]
    logger.info("\n%s", test_summary.to_string(index=False))

    (PROCESSED_DIR / "baseline_summary.json").write_text(
        json.dumps(
            {
                "feature_columns": len(feature_cols),
                "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
                "test_metrics": test_summary.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    logger.info(
        "Wrote %s and %s",
        PROCESSED_DIR / "baseline_results.csv",
        PROCESSED_DIR / "baseline_summary.json",
    )


if __name__ == "__main__":
    main()
