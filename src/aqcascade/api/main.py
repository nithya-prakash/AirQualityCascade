"""AirQualityCascade FastAPI app.

Run locally:
    uvicorn aqcascade.api.main:app --reload --port 8010

The model registry (trained models + feature table) is loaded once at
startup, not per-request -- see model_registry.py's docstring for why
predictions come from the last-ingested batch row rather than a live feed.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from aqcascade.api.model_registry import ModelRegistry, ModelUnavailableError, StationNotFoundError
from aqcascade.api.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)

registry: ModelRegistry | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global registry
    registry = ModelRegistry()
    yield


app = FastAPI(
    title="AirQualityCascade API",
    description="Germany-focused short-term PM2.5/NO2 forecasting. "
    "Predictions are computed from batch-ingested UBA/DWD data, not a live sensor feed "
    "-- see /health for the data's actual date range.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_registry() -> ModelRegistry:
    if registry is None:
        raise HTTPException(status_code=503, detail="Model registry not yet loaded")
    return registry


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    return get_registry().get_health()


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    responses={404: {"model": ErrorResponse}},
)
def model_info(pollutant: str = "pm25", horizon_hours: int = 1) -> dict:
    try:
        return get_registry().get_model_info(pollutant, horizon_hours)
    except ModelUnavailableError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/metrics")
def metrics() -> dict:
    return get_registry().get_metrics()


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def predict(request: PredictRequest) -> dict:
    reg = get_registry()
    try:
        return reg.predict(
            station_id=request.station_id,
            pollutant=request.pollutant,
            horizon_hours=request.horizon_hours,
            as_of=request.as_of,
        )
    except StationNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ModelUnavailableError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
