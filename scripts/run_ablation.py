"""Phase 8: the central experiment. Model A (local history) / B (+weather) /
C (+neighbor features) trained with a fixed architecture (XGBoost, the
strongest and fastest Phase 5 baseline) so the ONLY thing that changes
between them is feature access -- isolating what spatial context actually
contributes. Model D (the Phase 7 GNN) is pulled in from its own results
file rather than retrained here, since by construction its node features
already equal Model B's scope plus the graph (see
src/aqcascade/graph/snapshots.py) and it shares the identical dense
(station, hour) grid and time-split boundaries as A/B/C below.

Usage:
    python scripts/run_ablation.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from aqcascade.evaluation.metrics import compute_regression_metrics  # noqa: E402
from aqcascade.evaluation.split import chronological_split  # noqa: E402
from aqcascade.models.baselines import make_xgboost  # noqa: E402
from aqcascade.models.feature_groups import (  # noqa: E402
    MODEL_A_COLUMNS,
    MODEL_B_COLUMNS,
    MODEL_C_COLUMNS,
    validate_against,
)
from aqcascade.models.features import build_feature_matrix  # noqa: E402
from aqcascade.tracking.mlflow_utils import (  # noqa: E402
    configure_mlflow,
    dataset_version,
    log_split_metrics,
    log_split_periods,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

TARGETS = ["target_pm25_h1", "target_pm25_h2"]
TIERS = {"A": MODEL_A_COLUMNS, "B": MODEL_B_COLUMNS, "C": MODEL_C_COLUMNS}
RANDOM_SEED = 42


def main() -> None:
    configure_mlflow("aqcascade-ablation")
    logger.info("Loading features...")
    features_path = PROCESSED_DIR / "features.parquet"
    features = pd.read_parquet(features_path)
    ds_version = dataset_version(features_path)
    validate_against(features.columns.tolist())

    train, val, test = chronological_split(features, "timestamp", train_frac=0.7, val_frac=0.15)
    logger.info(
        "Split: train=%d (%s -> %s), test=%d (%s -> %s) [identical boundaries to Model D]",
        len(train),
        train.timestamp.min(),
        train.timestamp.max(),
        len(test),
        test.timestamp.min(),
        test.timestamp.max(),
    )

    results = []
    for target_col in TARGETS:
        train_t = train.dropna(subset=[target_col])
        val_t = val.dropna(subset=[target_col])
        test_t = test.dropna(subset=[target_col])

        for tier_name, feature_cols in TIERS.items():
            logger.info(
                "=== Target %s, Model %s (%d features) ===",
                target_col,
                tier_name,
                len(feature_cols),
            )
            X_train = build_feature_matrix(train_t, feature_cols)
            X_val = build_feature_matrix(val_t, feature_cols)
            X_test = build_feature_matrix(test_t, feature_cols)
            y_train, y_val, y_test = train_t[target_col], val_t[target_col], test_t[target_col]

            with mlflow.start_run(run_name=f"{target_col}__tier_{tier_name}"):
                mlflow.log_params(
                    {
                        "model_type": "xgboost",
                        "tier": tier_name,
                        "target": target_col,
                        "n_features": len(feature_cols),
                        "dataset_version": ds_version,
                        "random_seed": RANDOM_SEED,
                    }
                )
                log_split_periods(train_t, val_t, test_t)

                model = make_xgboost()
                t0 = time.time()
                model.fit(X_train, y_train)
                fit_seconds = time.time() - t0
                mlflow.log_metric("fit_seconds", fit_seconds)

                for split_name, X_split, y_split in [
                    ("train", X_train, y_train),
                    ("val", X_val, y_val),
                    ("test", X_test, y_test),
                ]:
                    preds = model.predict(X_split)
                    metrics = compute_regression_metrics(y_split.to_numpy(), preds)
                    log_split_metrics(split_name, metrics)
                    results.append(
                        {
                            "target": target_col,
                            "model": f"Model {tier_name} (XGBoost, {len(feature_cols)} features)",
                            "tier": tier_name,
                            "split": split_name,
                            "fit_seconds": round(fit_seconds, 2),
                            **metrics,
                        }
                    )
            logger.info("  fit in %.1fs", fit_seconds)

    results_df = pd.DataFrame(results)
    results_df.to_csv(PROCESSED_DIR / "ablation_results.csv", index=False)

    test_summary = results_df[results_df["split"] == "test"][
        ["target", "tier", "mae", "rmse", "r2", "n"]
    ]
    logger.info("\n%s", test_summary.to_string(index=False))

    (PROCESSED_DIR / "ablation_summary.json").write_text(
        json.dumps(
            {
                "tiers": {k: len(v) for k, v in TIERS.items()},
                "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
                "test_metrics": test_summary.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    logger.info(
        "Wrote %s and %s",
        PROCESSED_DIR / "ablation_results.csv",
        PROCESSED_DIR / "ablation_summary.json",
    )


if __name__ == "__main__":
    main()
