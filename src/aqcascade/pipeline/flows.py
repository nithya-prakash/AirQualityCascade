"""Prefect orchestration over this project's existing pipeline scripts.

Each stage (ingestion, validation+preprocessing, feature generation,
training, evaluation, model registration/artifact creation -- the seven
stages the brief asks for, with validation folded into preprocessing
exactly as Phase 3 built it and model registration realized as MLflow's
artifact logging inside each training script) is wrapped as a Prefect task
that invokes the real script, not a reimplementation of its logic. This
keeps one source of truth per pipeline stage (the script itself, already
tested and already run for real) while adding retry/observability/DAG
structure on top.

Runs entirely locally via `prefect flow run` or by calling `full_pipeline()`
directly -- no Prefect server, database, or deployment required, per the
brief's "avoid unnecessary infrastructure" instruction applied the same way
it's applied to Docker/CI elsewhere in this project.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from prefect import flow, task

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPTS_DIR = ROOT / "scripts"


def _run_script(name: str, *args: str) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / name), *args],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{name} exited with code {result.returncode}")


@task(name="ingest", retries=1, retry_delay_seconds=30)
def ingest_task(days: int = 90) -> None:
    """Stage 1: ingestion (UBA + DWD -> data/raw/)."""
    _run_script("ingest.py", "--days", str(days))


@task(name="validate_and_preprocess")
def preprocess_task() -> None:
    """Stages 2-3: schema validation (pandera) + cleaning + spatial/temporal
    alignment -> data/processed/panel.parquet. Validation is folded into
    this stage exactly as Phase 3 implemented it (validate-then-clean in
    one script), not artificially split into a separate flow step."""
    _run_script("preprocess.py")


@task(name="build_features")
def build_features_task() -> None:
    """Stage 4: leakage-safe feature engineering -> data/processed/features.parquet."""
    _run_script("build_features.py")


@task(name="train_baselines")
def train_baselines_task() -> None:
    """Stage 5 (part 1): Persistence/LR/RF/XGBoost, logged to MLflow."""
    _run_script("train_baselines.py")


@task(name="train_transformer")
def train_transformer_task() -> None:
    """Stage 5 (part 2): Temporal Transformer, logged to MLflow."""
    _run_script("train_transformer.py")


@task(name="train_gnn")
def train_gnn_task() -> None:
    """Stage 5 (part 3): spatial GNN (Model D), logged to MLflow."""
    _run_script("train_gnn.py")


@task(name="run_ablation")
def run_ablation_task() -> None:
    """Stage 5 (part 4) + stage 6 (evaluation): the controlled A/B/C spatial
    ablation -- the central experiment -- logged to MLflow."""
    _run_script("run_ablation.py")


@task(name="error_analysis")
def error_analysis_task() -> None:
    """Stage 6: error analysis + spike classification."""
    _run_script("error_analysis.py")


@flow(name="aqcascade-pipeline", log_prints=True)
def full_pipeline(days: int = 90, skip_ingestion: bool = True) -> None:
    """The full pipeline, ingestion through evaluation.

    `skip_ingestion` defaults to True: re-pulling 90 days from the live
    UBA/DWD APIs on every pipeline run isn't necessary once the data exists
    locally, and doing it by default would make this flow depend on
    external network availability for no benefit -- pass
    `skip_ingestion=False` explicitly to actually re-ingest.
    """
    if not skip_ingestion:
        ingest_task(days)
    preprocess_task()
    build_features_task()
    train_baselines_task()
    train_transformer_task()
    train_gnn_task()
    run_ablation_task()
    error_analysis_task()


if __name__ == "__main__":
    full_pipeline()
