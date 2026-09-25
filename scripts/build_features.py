"""Phase 4: leakage-safe feature engineering -> data/processed/features.parquet.

Usage:
    python scripts/build_features.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from aqcascade.features.build import build_feature_table  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
CONFIG_PATH = ROOT / "configs" / "features.yaml"


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())

    logger.info("Loading processed panel + station metadata...")
    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")
    station_meta = pd.read_parquet(PROCESSED_DIR / "station_metadata.parquet")

    logger.info("Building feature table (regularizing to hourly grid: %d stations)...",
                station_meta["station_id"].nunique())
    features, neighbor_graph = build_feature_table(panel, station_meta, config)

    feature_cols = [
        c for c in features.columns
        if c not in {"station_id", "timestamp", "latitude", "longitude"}
        and not c.startswith("target_")
    ]

    report = {
        "rows": int(len(features)),
        "stations": int(features["station_id"].nunique()),
        "n_feature_columns": len(feature_cols),
        "neighbor_graph_edges": int(len(neighbor_graph)),
        "neighbor_graph_stations_with_zero_neighbors": int(
            station_meta["station_id"].nunique() - neighbor_graph["station_id"].nunique()
        ),
        "target_columns": [c for c in features.columns if c.startswith("target_")],
        "target_missing_pct": {
            c: round(float(features[c].isna().mean()) * 100, 2)
            for c in features.columns if c.startswith("target_")
        },
        "feature_missing_pct_top10": (
            features[feature_cols].isna().mean().sort_values(ascending=False).head(10) * 100
        ).round(2).to_dict(),
    }

    features.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    neighbor_graph.to_parquet(PROCESSED_DIR / "neighbor_graph.parquet", index=False)
    report_path = PROCESSED_DIR / "features_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    logger.info("Wrote %s (%d rows, %d feature columns)",
                PROCESSED_DIR / "features.parquet", report["rows"], report["n_feature_columns"])
    logger.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
