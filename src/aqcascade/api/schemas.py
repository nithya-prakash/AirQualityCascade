"""Pydantic request/response models for the API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    station_id: int = Field(..., description="UBA station id, e.g. 21")
    pollutant: Literal["pm25", "no2"] = Field(
        default="pm25", description="Which pollutant to forecast"
    )
    horizon_hours: Literal[1, 2] = Field(
        default=1, description="Forecast horizon: 1h or 2h ahead (see README horizon note)"
    )
    as_of: datetime | None = Field(
        default=None,
        description="Feature cutoff time to predict from. Defaults to the latest hour "
        "this project has ingested data for. Must match an hour actually present in "
        "data/processed/features.parquet.",
    )


class TestMetrics(BaseModel):
    mae: float
    rmse: float
    r2: float
    note: str


class PredictResponse(BaseModel):
    station_id: int
    station_name: str
    pollutant: str
    unit: str
    as_of_timestamp: datetime
    prediction_timestamp: datetime
    horizon_hours: int
    predicted_value: float
    model_version: str
    uncertainty: TestMetrics | None = Field(
        default=None,
        description="Historical test-set residual spread for this model/horizon -- "
        "NOT a per-prediction confidence interval (this model doesn't produce one; "
        "see README Limitations).",
    )


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    n_stations: int
    data_as_of: str
    data_source: str


class ModelInfoResponse(BaseModel):
    model_version: str
    pollutant: str
    horizon_hours: int
    target_column: str
    architecture: str
    n_features: int
    n_estimators: int
    max_depth: int
    test_metrics: TestMetrics | None = None


class ErrorResponse(BaseModel):
    detail: str
