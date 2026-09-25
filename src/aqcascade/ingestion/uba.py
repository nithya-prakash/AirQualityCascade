"""Ingestion of the UBA (Umweltbundesamt) Air Data API v3.

No API key is required — see configs/data_sources.yaml for the endpoints
this module uses and how each one was verified during Phase 1 inspection.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from collections.abc import Iterable

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://www.umweltbundesamt.de/api/air_data/v3"

# Component ids confirmed against /components/json (Phase 1).
COMPONENTS = {
    "PM10": 1,
    "CO": 2,
    "O3": 3,
    "SO2": 4,
    "NO2": 5,
    "PM2.5": 9,
}

# Scope ids confirmed against /scopes/json (Phase 1). We only use the
# one-hour average — it's the finest resolution UBA actually publishes.
SCOPE_HOURLY = 2

STATION_COLUMNS = [
    "station_id",
    "station_code",
    "station_name",
    "station_city",
    "station_synonym",
    "active_from",
    "active_to",
    "longitude",
    "latitude",
    "network_id",
    "station_setting_id",
    "station_type_id",
    "network_code",
    "network_name",
    "station_setting_name",
    "station_setting_short_name",
    "station_type_name",
    "street",
    "street_nr",
    "zip_code",
]

_session = requests.Session()
_session.headers.update({"User-Agent": "AirQualityCascade/0.1 (portfolio project)"})


def _get_json(endpoint: str, params: dict) -> dict:
    resp = _session.get(f"{BASE_URL}/{endpoint}", params={**params, "lang": "en"}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_stations() -> pd.DataFrame:
    """Full UBA station registry (all networks, all time periods)."""
    payload = _get_json("stations/json", {"index": "id"})
    rows = [values for values in payload["data"].values()]
    df = pd.DataFrame(rows, columns=STATION_COLUMNS)
    df["station_id"] = df["station_id"].astype(int)
    df["longitude"] = df["longitude"].astype(float)
    df["latitude"] = df["latitude"].astype(float)
    df["active_from"] = pd.to_datetime(df["active_from"], errors="coerce")
    df["active_to"] = pd.to_datetime(df["active_to"], errors="coerce")
    return df


def fetch_reporting_station_ids(component_id: int, on_date: dt.date) -> set[int]:
    """Station ids that actually published a value for `component_id` on `on_date`.

    UBA's /stations endpoint does not support filtering by component (verified
    empirically — the `component` query param is silently ignored), so the
    reliable way to find which stations measure a pollutant is to query
    /measures for one day with no station filter and see who reported.
    """
    payload = _get_json(
        "measures/json",
        {
            "date_from": on_date.isoformat(),
            "time_from": "1",
            "date_to": on_date.isoformat(),
            "time_to": "24",
            "component": str(component_id),
            "scope": str(SCOPE_HOURLY),
        },
    )
    return {int(sid) for sid in payload.get("data", {}).keys()}


def _daterange_chunks(
    date_from: dt.date, date_to: dt.date, chunk_days: int = 7
) -> Iterable[tuple[dt.date, dt.date]]:
    cur = date_from
    while cur <= date_to:
        chunk_end = min(cur + dt.timedelta(days=chunk_days - 1), date_to)
        yield cur, chunk_end
        cur = chunk_end + dt.timedelta(days=1)


def fetch_measures(
    component_id: int,
    date_from: dt.date,
    date_to: dt.date,
    scope: int = SCOPE_HOURLY,
    chunk_days: int = 7,
    sleep_seconds: float = 0.3,
) -> pd.DataFrame:
    """All stations' hourly measurements for one component over a date range.

    Queried with no station filter (returns every reporting station at once)
    and chunked into `chunk_days`-day windows to keep each response a
    reasonable size, since UBA imposes no documented pagination.
    """
    frames = []
    chunks = list(_daterange_chunks(date_from, date_to, chunk_days))
    for i, (chunk_from, chunk_to) in enumerate(chunks):
        logger.info(
            "UBA component=%s chunk %d/%d: %s -> %s", component_id, i + 1, len(chunks),
            chunk_from, chunk_to,
        )
        payload = _get_json(
            "measures/json",
            {
                "date_from": chunk_from.isoformat(),
                "time_from": "1",
                "date_to": chunk_to.isoformat(),
                "time_to": "24",
                "component": str(component_id),
                "scope": str(scope),
            },
        )
        for station_id, records in payload.get("data", {}).items():
            for ts_start, rec in records.items():
                # rec = [component_id, scope_id, value, ts_end, quality_index]
                frames.append(
                    {
                        "station_id": int(station_id),
                        "component_id": int(rec[0]),
                        "scope_id": int(rec[1]),
                        "value": float(rec[2]) if rec[2] is not None else None,
                        "ts_start": ts_start,
                        "ts_end": rec[3],
                        "quality_index": rec[4],
                    }
                )
        if i + 1 < len(chunks):
            time.sleep(sleep_seconds)

    df = pd.DataFrame(frames)
    if df.empty:
        return df
    df["ts_start"] = pd.to_datetime(df["ts_start"])
    df["ts_end"] = df["ts_end"].apply(_parse_uba_timestamp)
    return df


def _parse_uba_timestamp(value: str) -> pd.Timestamp:
    """Parse UBA timestamps, handling the "24:00:00 == next day 00:00:00" convention."""
    if value.endswith("24:00:00"):
        date_part = value.split(" ")[0]
        return pd.Timestamp(date_part) + pd.Timedelta(days=1)
    return pd.Timestamp(value)
