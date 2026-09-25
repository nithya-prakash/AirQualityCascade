"""Phase 3: validate + clean + spatially/temporally align raw data -> data/processed/panel.parquet.

Usage:
    python scripts/preprocess.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402
from pandera.errors import SchemaErrors  # noqa: E402

from aqcascade.preprocessing import align, clean  # noqa: E402
from aqcascade.validation import schemas  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def validate_or_report(df: pd.DataFrame, schema, name: str) -> None:
    try:
        schema.validate(df, lazy=True)
        logger.info("%s: schema PASSED (%d rows)", name, len(df))
    except SchemaErrors as exc:
        logger.error("%s: schema FAILED\n%s", name, exc.failure_cases.head(20))
        raise


def main() -> None:
    report: dict = {}
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw data...")
    pm25_meta = pd.read_parquet(RAW_DIR / "uba" / "stations_pm25.parquet")
    pm25_measures = pd.read_parquet(RAW_DIR / "uba" / "measures_pm25.parquet")
    no2_measures = pd.read_parquet(RAW_DIR / "uba" / "measures_no2.parquet")
    weather = pd.read_parquet(RAW_DIR / "dwd" / "weather_hourly.parquet")
    join_df = pd.read_parquet(RAW_DIR / "dwd" / "uba_dwd_station_join.parquet")

    # ---- Validate raw inputs against physical-plausibility schemas ----
    validate_or_report(pm25_meta, schemas.uba_station_schema, "UBA station metadata")
    validate_or_report(pm25_measures, schemas.uba_measures_schema, "PM2.5 raw measures")
    validate_or_report(no2_measures, schemas.uba_measures_schema, "NO2 raw measures")
    validate_or_report(weather, schemas.dwd_weather_schema, "DWD weather")

    # ---- Clean: dedupe + drop impossible timestamps ----
    window_min, window_max = pm25_measures["ts_start"].min(), pm25_measures["ts_start"].max()

    pm25_measures, dup_pm25 = clean.deduplicate(pm25_measures, ["station_id", "ts_start"])
    no2_measures, dup_no2 = clean.deduplicate(no2_measures, ["station_id", "ts_start"])
    weather, dup_weather = clean.deduplicate(weather, ["dwd_station_id", "timestamp", "parameter"])

    pm25_measures, bad_ts_pm25 = clean.drop_invalid_timestamps(
        pm25_measures, "ts_start", window_min, window_max
    )
    no2_measures, bad_ts_no2 = clean.drop_invalid_timestamps(
        no2_measures, "ts_start", window_min, window_max
    )

    report["duplicates_dropped"] = {
        "pm25": dup_pm25,
        "no2": dup_no2,
        "weather": dup_weather,
    }
    report["invalid_timestamps_dropped"] = {"pm25": bad_ts_pm25, "no2": bad_ts_no2}

    # ---- Align: build the station-hour panel ----
    logger.info("Building aligned panel...")
    panel = align.build_panel(pm25_measures, no2_measures, weather, join_df, pm25_meta)

    # Flag (never drop) statistical outliers for downstream error analysis.
    panel["pm25_is_outlier"] = clean.flag_outliers_iqr(panel, "pm25")
    report["pm25_outliers_flagged"] = int(panel["pm25_is_outlier"].sum())

    validate_or_report(panel, schemas.panel_schema, "Aligned panel")

    # ---- Missing-data report (never filled — see module docstring) ----
    report["panel_rows"] = int(len(panel))
    report["panel_stations"] = int(panel["station_id"].nunique())
    report["missing_pct"] = {
        col: round(float(panel[col].isna().mean()) * 100, 2)
        for col in [
            "pm25",
            "no2",
            "temperature_c",
            "humidity_pct",
            "wind_speed_ms",
            "wind_direction_deg",
            "pressure_msl_hpa",
            "precipitation_mm",
        ]
        if col in panel.columns
    }

    panel.to_parquet(PROCESSED_DIR / "panel.parquet", index=False)
    pm25_meta.to_parquet(PROCESSED_DIR / "station_metadata.parquet", index=False)

    report_path = PROCESSED_DIR / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(
        "Wrote %s and validation report to %s", PROCESSED_DIR / "panel.parquet", report_path
    )
    logger.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
