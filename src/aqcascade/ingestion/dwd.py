"""Ingestion of DWD (Deutscher Wetterdienst) Open Data hourly station files.

No API key is required — see configs/data_sources.yaml for the endpoints
this module uses and how each one was verified during Phase 1 inspection.
"""

from __future__ import annotations

import io
import logging
import zipfile
from collections.abc import Iterable

import pandas as pd
import requests

from aqcascade.common.geo import haversine_km

logger = logging.getLogger(__name__)

BASE_URL = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly"

# parameter folder -> (file code, value columns kept, rename map)
# Confirmed by downloading and inspecting one real "recent" zip per parameter
# during Phase 2 (see conversation / configs/data_sources.yaml).
PARAMETERS: dict[str, dict] = {
    "air_temperature": {
        "code": "TU",
        "columns": {"TT_TU": "temperature_c", "RF_TU": "humidity_pct"},
    },
    "wind": {
        "code": "FF",
        "columns": {"F": "wind_speed_ms", "D": "wind_direction_deg"},
    },
    "pressure": {
        "code": "P0",
        # DWD's hourly pressure product has two columns: "P" (mean sea-level
        # pressure) and "P0" (station-level pressure, NOT reduced). Verified
        # empirically during Phase 3: the "P0" column read ~700 hPa for the
        # Zugspitze station (2956m elevation) — physically correct for
        # station-level pressure at that altitude via the barometric
        # formula, and impossible for a sea-level-reduced value. "P0" is
        # also the file-naming code (stundenwerte_P0_*.zip), unrelated to
        # which internal column is which.
        "columns": {"P": "pressure_msl_hpa"},
    },
    "precipitation": {
        "code": "RR",
        "columns": {"R1": "precipitation_mm"},
    },
}

MISSING_SENTINEL = -999

_session = requests.Session()
_session.headers.update({"User-Agent": "AirQualityCascade/0.1 (portfolio project)"})

STATION_LIST_COLSPECS = [
    (0, 5),  # Stations_id
    (6, 14),  # von_datum
    (15, 23),  # bis_datum
    (24, 38),  # Stationshoehe
    (39, 50),  # geoBreite (lat)
    (51, 60),  # geoLaenge (lon)
    (61, 102),  # Stationsname
    (102, 143),  # Bundesland
]
STATION_LIST_NAMES = [
    "station_id",
    "von_datum",
    "bis_datum",
    "height_m",
    "latitude",
    "longitude",
    "station_name",
    "bundesland",
]


def fetch_station_list(parameter: str, recent: bool = True) -> pd.DataFrame:
    """Station metadata (id, lat/lon, active period) for a DWD parameter."""
    code = PARAMETERS[parameter]["code"]
    folder = "recent" if recent else "historical"
    url = f"{BASE_URL}/{parameter}/{folder}/{code}_Stundenwerte_Beschreibung_Stationen.txt"
    resp = _session.get(url, timeout=60)
    resp.raise_for_status()
    text = resp.content.decode("latin-1")
    lines = text.splitlines()[2:]  # skip header + dashed separator
    buf = io.StringIO("\n".join(line for line in lines if line.strip()))
    df = pd.read_fwf(buf, colspecs=STATION_LIST_COLSPECS, names=STATION_LIST_NAMES, dtype=str)
    df["station_id"] = df["station_id"].str.strip().astype(int)
    df["von_datum"] = pd.to_datetime(df["von_datum"].str.strip(), format="%Y%m%d")
    df["bis_datum"] = pd.to_datetime(df["bis_datum"].str.strip(), format="%Y%m%d")
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    df["height_m"] = df["height_m"].astype(float)
    df["station_name"] = df["station_name"].str.strip()
    df["bundesland"] = df["bundesland"].str.strip()
    return df


def find_nearest_station(
    lat: float, lon: float, dwd_stations: pd.DataFrame, currently_active_only: bool = True
) -> tuple[int, float]:
    """Nearest DWD station (by haversine distance) to a (lat, lon) point.

    Returns (dwd_station_id, distance_km). This is an explicit spatial join
    between two different physical networks (UBA pollution stations, DWD
    weather stations) — never treated as a co-located sensor.
    """
    candidates = dwd_stations
    if currently_active_only:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
        candidates = candidates[candidates["bis_datum"] >= cutoff]
    if candidates.empty:
        raise ValueError("No active DWD stations available for nearest-station lookup")
    distances = candidates.apply(
        lambda row: haversine_km(lat, lon, row["latitude"], row["longitude"]), axis=1
    )
    idx = distances.idxmin()
    return int(candidates.loc[idx, "station_id"]), float(distances.loc[idx])


def download_recent_product(parameter: str, station_id: int) -> pd.DataFrame:
    """Download + parse one DWD station's 'recent' (~500 day) hourly file."""
    code = PARAMETERS[parameter]["code"]
    sid_str = f"{station_id:05d}"
    url = f"{BASE_URL}/{parameter}/recent/stundenwerte_{code}_{sid_str}_akt.zip"
    resp = _session.get(url, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        product_name = next(n for n in zf.namelist() if n.startswith("produkt_"))
        with zf.open(product_name) as f:
            df = pd.read_csv(f, sep=";", skipinitialspace=True)

    df.columns = [c.strip() for c in df.columns]
    df["MESS_DATUM"] = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d%H")
    value_cols = PARAMETERS[parameter]["columns"]
    keep = ["STATIONS_ID", "MESS_DATUM", *value_cols.keys()]
    df = df[[c for c in keep if c in df.columns]].rename(
        columns={"STATIONS_ID": "dwd_station_id", "MESS_DATUM": "timestamp", **value_cols}
    )
    return _replace_missing_sentinel(df, value_cols.values())


def _replace_missing_sentinel(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """DWD encodes missing readings as -999 rather than an empty field."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace(MISSING_SENTINEL, pd.NA)
    return df
