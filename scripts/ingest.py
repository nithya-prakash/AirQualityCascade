"""Phase 2 ingestion orchestration: UBA pollution + DWD weather -> data/raw/.

Usage:
    python scripts/ingest.py                 # full 90-day pull
    python scripts/ingest.py --days 3 --dry-run   # quick smoke test
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from aqcascade.ingestion import dwd, uba  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DWD_PARAMETERS = ["air_temperature", "wind", "pressure", "precipitation"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90, help="history window length in days")
    ap.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="last full day to pull (YYYY-MM-DD); defaults to yesterday",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="limit to a handful of stations to sanity-check the pipeline fast",
    )
    ap.add_argument(
        "--max-stations",
        type=int,
        default=None,
        help="cap the number of UBA stations processed (mainly for --dry-run)",
    )
    args = ap.parse_args()

    end_date = (
        dt.date.fromisoformat(args.end_date)
        if args.end_date
        else dt.date.today() - dt.timedelta(days=1)
    )
    pull_days = 3 if args.dry_run else args.days
    pull_start = end_date - dt.timedelta(days=pull_days - 1)
    logger.info("Ingestion window: %s -> %s (%d days)", pull_start, end_date, pull_days)

    (DATA_DIR / "uba").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "dwd").mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "ingested_at": dt.datetime.now(dt.UTC).isoformat(),
        "window_start": pull_start.isoformat(),
        "window_end": end_date.isoformat(),
        "window_days": pull_days,
        "dry_run": args.dry_run,
    }

    # ---- UBA: station registry + reporting-station discovery ----
    logger.info("Fetching UBA station registry...")
    all_stations = uba.fetch_stations()
    all_stations.to_parquet(DATA_DIR / "uba" / "stations_all.parquet", index=False)
    manifest["uba_stations_total_ever"] = int(len(all_stations))

    logger.info("Discovering stations currently reporting PM2.5...")
    pm25_ids = uba.fetch_reporting_station_ids(uba.COMPONENTS["PM2.5"], end_date)
    logger.info("Discovering stations currently reporting NO2...")
    no2_ids = uba.fetch_reporting_station_ids(uba.COMPONENTS["NO2"], end_date)

    if args.max_stations:
        pm25_ids = set(sorted(pm25_ids)[: args.max_stations])
        no2_ids = no2_ids & pm25_ids

    manifest["pm25_reporting_stations"] = len(pm25_ids)
    manifest["no2_reporting_stations"] = len(no2_ids)
    logger.info("PM2.5 stations: %d | NO2 stations: %d", len(pm25_ids), len(no2_ids))

    pm25_station_meta = all_stations[all_stations["station_id"].isin(pm25_ids)].copy()
    pm25_station_meta.to_parquet(DATA_DIR / "uba" / "stations_pm25.parquet", index=False)

    # ---- UBA: historical measures ----
    chunk_days = 2 if args.dry_run else 7

    logger.info("Pulling PM2.5 measures (%s -> %s)...", pull_start, end_date)
    pm25_measures = uba.fetch_measures(
        uba.COMPONENTS["PM2.5"], pull_start, end_date, chunk_days=chunk_days
    )
    pm25_measures = pm25_measures[pm25_measures["station_id"].isin(pm25_ids)]
    pm25_measures.to_parquet(DATA_DIR / "uba" / "measures_pm25.parquet", index=False)
    manifest["pm25_rows"] = int(len(pm25_measures))

    logger.info("Pulling NO2 measures (%s -> %s)...", pull_start, end_date)
    no2_measures = uba.fetch_measures(
        uba.COMPONENTS["NO2"], pull_start, end_date, chunk_days=chunk_days
    )
    no2_measures = no2_measures[no2_measures["station_id"].isin(no2_ids)]
    no2_measures.to_parquet(DATA_DIR / "uba" / "measures_no2.parquet", index=False)
    manifest["no2_rows"] = int(len(no2_measures))

    # ---- DWD: nearest-station join + hourly weather ----
    logger.info("Fetching DWD station lists for %s...", DWD_PARAMETERS)
    dwd_station_lists = {p: dwd.fetch_station_list(p) for p in DWD_PARAMETERS}

    join_rows = []
    dwd_cache: dict[tuple[str, int], pd.DataFrame] = {}

    pm25_stations_iter = pm25_station_meta.to_dict("records")
    logger.info("Resolving nearest DWD station per parameter for %d UBA stations...", len(
        pm25_stations_iter
    ))
    for i, station in enumerate(pm25_stations_iter):
        for param in DWD_PARAMETERS:
            try:
                dwd_id, dist_km = dwd.find_nearest_station(
                    station["latitude"], station["longitude"], dwd_station_lists[param]
                )
            except ValueError:
                logger.warning(
                    "No active DWD station for parameter=%s near UBA station %s",
                    param, station["station_id"],
                )
                continue
            join_rows.append(
                {
                    "uba_station_id": station["station_id"],
                    "parameter": param,
                    "dwd_station_id": dwd_id,
                    "distance_km": dist_km,
                }
            )
            key = (param, dwd_id)
            if key not in dwd_cache:
                try:
                    df = dwd.download_recent_product(param, dwd_id)
                    df["parameter"] = param
                    dwd_cache[key] = df
                    time.sleep(0.15)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to download DWD %s station %s: %s", param, dwd_id, e)
        if (i + 1) % 50 == 0:
            logger.info("  ...%d/%d UBA stations resolved", i + 1, len(pm25_stations_iter))

    join_df = pd.DataFrame(join_rows)
    join_df.to_parquet(DATA_DIR / "dwd" / "uba_dwd_station_join.parquet", index=False)

    weather_frames = list(dwd_cache.values())
    if weather_frames:
        weather_df = pd.concat(weather_frames, ignore_index=True)
        window_start_ts = pd.Timestamp(pull_start)
        window_end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        weather_df = weather_df[
            (weather_df["timestamp"] >= window_start_ts)
            & (weather_df["timestamp"] < window_end_ts)
        ]
        weather_df.to_parquet(DATA_DIR / "dwd" / "weather_hourly.parquet", index=False)
        manifest["dwd_unique_stations_downloaded"] = len(dwd_cache)
        manifest["dwd_weather_rows"] = int(len(weather_df))
    else:
        manifest["dwd_unique_stations_downloaded"] = 0
        manifest["dwd_weather_rows"] = 0

    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Manifest written to %s", manifest_path)
    logger.info(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
