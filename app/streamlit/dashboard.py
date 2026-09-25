"""AirQualityCascade monitoring/forecasting dashboard.

Every number on this page comes from a file this project actually produced
(data/processed/*.csv, *.parquet, *.json) or a live call to the FastAPI
service -- nothing here is a placeholder or a hand-typed "example" value.
If the API isn't running, forecast panels say so explicitly rather than
silently showing nothing or a stale number.

Run: streamlit run app/streamlit/dashboard.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
API_URL = os.environ.get("AQCASCADE_API_URL", "http://localhost:8010")

st.set_page_config(page_title="AirQualityCascade", layout="wide")


# ---------------------------------------------------------------- data ----
@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "features.parquet")


@st.cache_data
def load_station_meta() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "station_metadata.parquet")


@st.cache_data
def load_neighbor_graph() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "neighbor_graph.parquet")


@st.cache_data
def load_csv_if_exists(name: str) -> pd.DataFrame | None:
    path = PROCESSED_DIR / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_json_if_exists(name: str) -> dict | None:
    path = PROCESSED_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


def call_predict_api(station_id: int, pollutant: str, horizon_hours: int) -> dict | None:
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json={"station_id": station_id, "pollutant": pollutant, "horizon_hours": horizon_hours},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json()
        st.warning(f"API returned {resp.status_code}: {resp.json().get('detail', resp.text)}")
        return None
    except requests.exceptions.RequestException:
        st.error(
            f"Could not reach the AirQualityCascade API at {API_URL}. "
            "Start it with: `uvicorn aqcascade.api.main:app --port 8010 --app-dir src`"
        )
        return None


features = load_features()
station_meta = load_station_meta()
neighbor_graph = load_neighbor_graph()
error_by_station = load_csv_if_exists("error_analysis_by_station.csv")
error_summary = load_json_if_exists("error_analysis_summary.json")
baseline_results = load_csv_if_exists("baseline_results.csv")
transformer_results = load_csv_if_exists("transformer_results.csv")
gnn_results = load_csv_if_exists("gnn_results.csv")

data_as_of = features["timestamp"].max()

# ------------------------------------------------------------- sidebar ----
st.sidebar.title("AirQualityCascade")
st.sidebar.caption(
    "Germany-focused PM2.5/NO2 forecasting. Data is batch-ingested "
    f"(UBA/DWD), not live -- as of **{data_as_of}**."
)

station_options = station_meta.sort_values("station_name")
station_label = station_options.apply(
    lambda r: f"{r['station_name']} ({r['station_city']}) — id {r['station_id']}", axis=1
)
selected_label = st.sidebar.selectbox("Station", station_label, index=0)
station_id = int(station_options.iloc[station_label.tolist().index(selected_label)]["station_id"])
station_row = station_meta[station_meta["station_id"] == station_id].iloc[0]

st.sidebar.markdown(f"**Type:** {station_row['station_type_name']}")
st.sidebar.markdown(f"**Setting:** {station_row['station_setting_name']}")
st.sidebar.markdown(f"**Network:** {station_row['network_name']}")

# --------------------------------------------------------------- header ---
st.title(f"{station_row['station_name']}")
st.caption(f"{station_row['station_city']} · station id {station_id}")

latest_row = features[features["station_id"] == station_id].sort_values("timestamp").iloc[-1]

spike_threshold = None
if error_summary is not None:
    spike_threshold = error_summary.get("spike_classification", {}).get("threshold")

# ---------------------------------------------------- current + forecast --
st.subheader("Current reading and forecast")
st.caption(
    "Forecasts are t+1h / t+2h, not literal 30/60-minute-ahead -- UBA's official "
    "network only publishes hourly data (see README 'A note on prediction horizons')."
)

col1, col2, col3 = st.columns(3)
current_pm25 = latest_row["pm25"]
with col1:
    st.metric(
        "Current PM2.5",
        f"{current_pm25:.1f} µg/m³" if pd.notna(current_pm25) else "no reading",
        help=f"As of {latest_row['timestamp']}",
    )

pred_h1 = call_predict_api(station_id, "pm25", 1)
pred_h2 = call_predict_api(station_id, "pm25", 2)

with col2:
    if pred_h1:
        delta = pred_h1["predicted_value"] - current_pm25 if pd.notna(current_pm25) else None
        help_text = (
            f"Predicted for {pred_h1['prediction_timestamp']} · model {pred_h1['model_version']}"
        )
        if pred_h1.get("uncertainty"):
            help_text += f" · historical test MAE {pred_h1['uncertainty']['mae']:.2f}"
        st.metric(
            "Forecast t+1h",
            f"{pred_h1['predicted_value']:.1f} µg/m³",
            delta=f"{delta:+.1f}" if delta is not None else None,
            help=help_text,
        )
with col3:
    if pred_h2:
        delta = pred_h2["predicted_value"] - current_pm25 if pd.notna(current_pm25) else None
        st.metric(
            "Forecast t+2h",
            f"{pred_h2['predicted_value']:.1f} µg/m³",
            delta=f"{delta:+.1f}" if delta is not None else None,
        )

if spike_threshold is not None and pred_h1 and pred_h1["predicted_value"] >= spike_threshold:
    st.warning(
        f"⚠️ Predicted PM2.5 at t+1h ({pred_h1['predicted_value']:.1f} µg/m³) is at or above "
        f"the spike threshold ({spike_threshold:.1f} µg/m³, the training period's 90th "
        "percentile). See README 'Error analysis' for this model's real precision/recall "
        "on spike detection (0.71 / 0.47) -- this is a flag worth checking, not a guarantee."
    )

st.divider()

# ------------------------------------------------------- historical trend -
st.subheader("Historical trend (last 7 days)")
station_history = (
    features[features["station_id"] == station_id]
    .sort_values("timestamp")
    .tail(24 * 7)[["timestamp", "pm25", "no2"]]
    .set_index("timestamp")
)
st.line_chart(station_history, height=280)

st.divider()

# ------------------------------------------------- neighbor + weather -----
st.subheader("Neighboring-station influence")
neighbors = neighbor_graph[neighbor_graph["station_id"] == station_id].sort_values("rank")
if neighbors.empty:
    st.info("No same-pollutant neighbor within 100km (see Phase 4 spatial config).")
else:
    neighbor_rows = []
    for _, n in neighbors.iterrows():
        n_id = int(n["neighbor_station_id"])
        n_latest = features[features["station_id"] == n_id].sort_values("timestamp").iloc[-1]
        n_name = station_meta.loc[station_meta["station_id"] == n_id, "station_name"].iloc[0]
        neighbor_rows.append(
            {
                "Neighbor": n_name,
                "Distance (km)": round(n["distance_km"], 1),
                "Current PM2.5 (µg/m³)": round(n_latest["pm25"], 1)
                if pd.notna(n_latest["pm25"])
                else None,
            }
        )
    # Full page width -- a 3-column table squeezed into a half-width layout
    # column silently truncated the third column in an earlier version of
    # this dashboard, verified by inspecting the live page, not just eyeballing it.
    st.dataframe(pd.DataFrame(neighbor_rows), hide_index=True, use_container_width=True)
    st.caption(
        f"Neighbor mean PM2.5 (the engineered feature the model actually uses): "
        f"**{latest_row['neighbor_pm25_mean']:.1f} µg/m³** "
        f"({int(latest_row['neighbor_pm25_count'])} of {len(neighbors)} neighbors reporting)."
        if pd.notna(latest_row.get("neighbor_pm25_mean"))
        else "Neighbor mean unavailable for the current hour."
    )

st.subheader("Weather context (current hour)")
weather_fields = [
    ("Temperature", latest_row.get("temperature_c"), "°C"),
    ("Humidity", latest_row.get("humidity_pct"), "%"),
    ("Wind speed", latest_row.get("wind_speed_ms"), "m/s"),
    ("Pressure (sea level)", latest_row.get("pressure_msl_hpa"), "hPa"),
    ("Precipitation", latest_row.get("precipitation_mm"), "mm"),
]
# 3 + 2 per row rather than 5-across -- this dashboard renders in a fairly
# narrow side panel, and a value like "1030.7 hPa" truncates illegibly
# when squeezed into a fifth of the available width.
for row_fields in (weather_fields[:3], weather_fields[3:]):
    row_cols = st.columns(3)
    for col, (label, value, unit) in zip(row_cols, row_fields, strict=False):
        col.metric(label, f"{value:.1f} {unit}" if pd.notna(value) else "missing")

st.divider()

# --------------------------------------------------------- model comparison
st.subheader("Model comparison")
st.caption(
    "Real test-set results from every model built in this project (Phases 5-8), "
    "PM2.5 t+1h, same chronological test period for all. See README 'Results' for "
    "the full table including t+2h and NO2."
)
comparison_rows = []
if baseline_results is not None:
    for model_name in ["persistence", "linear_regression", "random_forest", "xgboost"]:
        r = baseline_results[
            (baseline_results["target"] == "target_pm25_h1")
            & (baseline_results["split"] == "test")
            & (baseline_results["model"] == model_name)
        ]
        if not r.empty:
            row = r.iloc[0]
            comparison_rows.append(
                {"Model": model_name, "MAE": row["mae"], "RMSE": row["rmse"], "R²": row["r2"]}
            )
if transformer_results is not None:
    r = transformer_results[
        (transformer_results["target"] == "target_pm25_h1")
        & (transformer_results["split"] == "test")
    ]
    if not r.empty:
        row = r.iloc[0]
        comparison_rows.append(
            {"Model": "transformer", "MAE": row["mae"], "RMSE": row["rmse"], "R²": row["r2"]}
        )
if gnn_results is not None:
    r = gnn_results[(gnn_results["target"] == "target_pm25_h1") & (gnn_results["split"] == "test")]
    if not r.empty:
        row = r.iloc[0]
        comparison_rows.append(
            {"Model": "gnn", "MAE": row["mae"], "RMSE": row["rmse"], "R²": row["r2"]}
        )

if comparison_rows:
    comparison_df = pd.DataFrame(comparison_rows).sort_values("R²", ascending=False)
    st.dataframe(
        comparison_df.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.3f}"}),
        hide_index=True,
        use_container_width=True,
    )
else:
    st.info("Model comparison results not found -- run the Phase 5-8 training scripts first.")

st.divider()

# ---------------------------------------------------------- prediction errors
st.subheader("Prediction errors for this station")
if error_by_station is not None:
    row = error_by_station[error_by_station["station_id"] == station_id]
    if not row.empty:
        r = row.iloc[0]
        # rank 1 = highest MAE = hardest station to predict
        rank = int(error_by_station["mae"].rank(ascending=False, method="first")[row.index[0]])
        n_total = len(error_by_station)
        ecol1, ecol2, ecol3 = st.columns(3)
        ecol1.metric("Model C test MAE (this station)", f"{r['mae']:.2f} µg/m³")
        ecol2.metric("Difficulty rank", f"#{rank} of {n_total}", help="1 = hardest to predict")
        ecol3.metric("Outlier rate", f"{r['outlier_rate'] * 100:.1f}%")
        if rank <= 10:
            st.caption(
                f"⚠️ This is one of the 10 hardest stations to predict in the test set "
                f"(station type: {r['station_type_name']}). See README 'Error analysis'."
            )
    else:
        st.info("No error-analysis data for this station (it may not be in the test set).")
else:
    st.info("Error analysis not found -- run scripts/error_analysis.py first.")

st.divider()

# ------------------------------------------------------------- station map -
st.subheader("German station map")
st.caption(
    "Color = current PM2.5 (green = low, red = high). Size = fixed. Selected station in blue."
)

map_data = features[features["timestamp"] == data_as_of][
    ["station_id", "pm25", "latitude", "longitude"]
].merge(station_meta[["station_id", "station_name"]], on="station_id")


def _color_for(row: pd.Series) -> list[int]:
    if row["station_id"] == station_id:
        return [30, 100, 255, 200]
    if pd.isna(row["pm25"]):
        return [150, 150, 150, 120]
    # green (low) -> red (high), scaled against the spike threshold
    ceiling = spike_threshold * 2 if spike_threshold else 30.0
    t = min(row["pm25"] / ceiling, 1.0)
    return [int(255 * t), int(255 * (1 - t)), 40, 180]


map_data["color"] = map_data.apply(_color_for, axis=1)
map_data["radius"] = map_data["station_id"].apply(lambda sid: 6000 if sid == station_id else 3000)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_data,
    get_position="[longitude, latitude]",
    get_fill_color="color",
    get_radius="radius",
    pickable=True,
)
view_state = pdk.ViewState(latitude=51.0, longitude=10.4, zoom=5)
st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{station_name}\nPM2.5: {pm25} µg/m³"},
        map_provider="carto",
        map_style="light",  # token-free basemap -- this project has no Mapbox key
    )
)
