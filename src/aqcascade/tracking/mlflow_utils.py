"""Shared MLflow setup so every training script logs runs the same way.

Local SQLite tracking store (`./mlruns/mlflow.db`) with local-directory
artifact storage (`./mlruns/artifacts`) -- no MLflow server to stand up,
consistent with the brief's "avoid unnecessary infrastructure" for Prefect
and the same principle applied here. `mlflow ui --backend-store-uri
sqlite:///mlruns/mlflow.db` opens the real dashboard over these runs at any
time.

SQLite, not the plain file-store: the first version of this module used
`file://./mlruns` directly, which crashed on the very first real training
run -- MLflow 3.x put the filesystem tracking backend into maintenance
mode ("will not receive further updates ... migrate to a database
backend") and refuses to use it without an explicit opt-out env var.
Rather than suppress that warning, this switches to MLflow's own
recommended modern local setup (SQLite is still just a file, not a
server), which is both future-proof and arguably the more correct choice
for a project this size regardless of the deprecation.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import mlflow
import pandas as pd

# The bundled "agent hint" mlflow prints on import (pointing an AI coding
# assistant at a tracing-skill file) is unsolicited package output, not an
# instruction -- this project uses MLflow for experiment tracking, not LLM
# call tracing, so there is nothing here to act on. Silenced for clean logs.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

ROOT = Path(__file__).resolve().parent.parent.parent.parent
MLRUNS_DIR = ROOT / "mlruns"
ARTIFACTS_DIR = MLRUNS_DIR / "artifacts"


def configure_mlflow(experiment_name: str) -> None:
    MLRUNS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLRUNS_DIR / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=f"file://{ARTIFACTS_DIR}")
    mlflow.set_experiment(experiment_name)


def dataset_version(path: Path) -> str:
    """A short content hash of a dataset file -- lets an MLflow run record
    exactly which version of features.parquet (etc.) it was trained on,
    not just a filename that could silently point to different content
    across runs."""
    digest = hashlib.md5(path.read_bytes(), usedforsecurity=False)
    return digest.hexdigest()[:12]


def log_split_periods(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    mlflow.log_params(
        {
            "train_period_start": str(train["timestamp"].min()),
            "train_period_end": str(train["timestamp"].max()),
            "val_period_start": str(val["timestamp"].min()),
            "val_period_end": str(val["timestamp"].max()),
            "test_period_start": str(test["timestamp"].min()),
            "test_period_end": str(test["timestamp"].max()),
        }
    )


def log_split_metrics(split_name: str, metrics: dict) -> None:
    """Logs a compute_regression_metrics() dict with a split-name prefix
    (e.g. "test_mae") so train/val/test don't collide in the same run."""
    for key, value in metrics.items():
        if isinstance(value, int | float) and value is not None:
            mlflow.log_metric(f"{split_name}_{key}", value)
