"""Phase 6: train the Temporal Transformer on PM2.5 t+1h/t+2h, evaluated on
the same chronological test period as the Phase 5 baselines.

Usage:
    python scripts/train_transformer.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from aqcascade.evaluation.metrics import compute_regression_metrics  # noqa: E402
from aqcascade.evaluation.split import chronological_split  # noqa: E402
from aqcascade.models.sequence_data import (  # noqa: E402
    SEQUENCE_FEATURE_COLUMNS,
    PollutionSequenceDataset,
    apply_normalization,
    build_samples,
    build_station_arrays,
    compute_normalization_stats,
)
from aqcascade.models.transformer import PollutionTransformer  # noqa: E402
from aqcascade.tracking.mlflow_utils import configure_mlflow, dataset_version  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models" / "transformer"

TARGET_COLS = ["target_pm25_h1", "target_pm25_h2"]
SEQ_LEN = 24
BATCH_SIZE = 1024
MAX_EPOCHS = 20
PATIENCE = 3
LEARNING_RATE = 1e-3
RANDOM_SEED = 42


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, device, target_mean, target_std, optimizer=None) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_norm = (y - target_mean) / target_std
        with torch.set_grad_enabled(is_train):
            pred = model(x)
            loss = torch.nn.functional.mse_loss(pred, y_norm)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
    return total_loss / total_n


@torch.no_grad()
def predict(model, loader, device, target_mean, target_std) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device)
        pred_norm = model(x).cpu().numpy()
        preds.append(pred_norm * target_std.numpy() + target_mean.numpy())
        targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    configure_mlflow("aqcascade-transformer")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = get_device()
    logger.info("Using device: %s", device)

    logger.info("Loading features and building sequence windows (seq_len=%d)...", SEQ_LEN)
    features_path = PROCESSED_DIR / "features.parquet"
    features = pd.read_parquet(features_path)
    ds_version = dataset_version(features_path)

    station_arrays, timestamps, channel_names = build_station_arrays(features)
    samples = build_samples(features, TARGET_COLS, SEQ_LEN)
    logger.info(
        "%d valid (station, time) samples out of %d panel rows", len(samples), len(features)
    )

    train_s, val_s, test_s = chronological_split(
        samples, "timestamp", train_frac=0.7, val_frac=0.15
    )
    logger.info(
        "Split: train=%d (%s -> %s), val=%d (%s -> %s), test=%d (%s -> %s)",
        len(train_s),
        train_s.timestamp.min(),
        train_s.timestamp.max(),
        len(val_s),
        val_s.timestamp.min(),
        val_s.timestamp.max(),
        len(test_s),
        test_s.timestamp.min(),
        test_s.timestamp.max(),
    )

    train_end = train_s["timestamp"].max()
    normalize_cols = [
        c
        for c in SEQUENCE_FEATURE_COLUMNS
        if c
        not in (
            "hour_sin",
            "hour_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "wind_direction_sin",
            "wind_direction_cos",
        )
    ]
    feature_stats = compute_normalization_stats(
        station_arrays, timestamps, channel_names, normalize_cols, train_end
    )
    logger.info("Feature normalization stats (train-only): %s", feature_stats)
    norm_arrays = apply_normalization(station_arrays, channel_names, normalize_cols, feature_stats)

    target_mean = torch.tensor(train_s[TARGET_COLS].to_numpy().mean(axis=0), dtype=torch.float32)
    target_std = torch.tensor(train_s[TARGET_COLS].to_numpy().std(axis=0), dtype=torch.float32)
    logger.info("Target normalization (train-only): mean=%s std=%s", target_mean, target_std)

    train_ds = PollutionSequenceDataset(train_s, norm_arrays, TARGET_COLS, SEQ_LEN)
    val_ds = PollutionSequenceDataset(val_s, norm_arrays, TARGET_COLS, SEQ_LEN)
    test_ds = PollutionSequenceDataset(test_s, norm_arrays, TARGET_COLS, SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = PollutionTransformer(
        n_channels=len(channel_names), n_horizons=len(TARGET_COLS), seq_len=SEQ_LEN
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", n_params)

    mlflow.start_run(run_name="transformer")
    mlflow.log_params(
        {
            "model_type": "transformer",
            "targets": ",".join(TARGET_COLS),
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "random_seed": RANDOM_SEED,
            "n_channels": len(channel_names),
            "n_parameters": n_params,
            "dataset_version": ds_version,
            "device": str(device),
        }
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    t_start = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        train_loss = run_epoch(
            model, train_loader, device, target_mean.to(device), target_std.to(device), optimizer
        )
        val_loss = run_epoch(
            model, val_loader, device, target_mean.to(device), target_std.to(device)
        )
        epoch_s = time.time() - t0
        logger.info(
            "Epoch %d: train_loss=%.4f val_loss=%.4f (%.1fs)", epoch, train_loss, val_loss, epoch_s
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODELS_DIR / "best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                logger.info(
                    "Early stopping at epoch %d (no val improvement for %d epochs)", epoch, PATIENCE
                )
                break

    total_train_s = time.time() - t_start
    logger.info("Training took %.1fs total", total_train_s)

    model.load_state_dict(torch.load(MODELS_DIR / "best_model.pt", weights_only=True))

    results = []
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        preds, targets = predict(model, loader, device, target_mean, target_std)
        for i, target_col in enumerate(TARGET_COLS):
            metrics = compute_regression_metrics(targets[:, i], preds[:, i])
            results.append(
                {"target": target_col, "model": "transformer", "split": split_name, **metrics}
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(PROCESSED_DIR / "transformer_results.csv", index=False)
    logger.info(
        "\n%s",
        results_df[results_df.split == "test"][["target", "mae", "rmse", "r2", "n"]].to_string(
            index=False
        ),
    )

    mlflow.log_metric("total_train_seconds", total_train_s)
    mlflow.log_metric("epochs_trained", len(history))
    for _, row in results_df.iterrows():
        for metric_name in ("mae", "rmse", "r2"):
            mlflow.log_metric(f"{row['split']}_{row['target']}_{metric_name}", row[metric_name])
    mlflow.log_artifact(str(MODELS_DIR / "best_model.pt"))

    summary = {
        "device": str(device),
        "seq_len": SEQ_LEN,
        "n_channels": len(channel_names),
        "channel_names": channel_names,
        "n_parameters": n_params,
        "epochs_trained": len(history),
        "total_train_seconds": round(total_train_s, 1),
        "split_sizes": {"train": len(train_s), "val": len(val_s), "test": len(test_s)},
        "training_history": history,
        "test_metrics": results_df[results_df.split == "test"].to_dict(orient="records"),
    }
    (PROCESSED_DIR / "transformer_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "Wrote %s and %s",
        PROCESSED_DIR / "transformer_results.csv",
        PROCESSED_DIR / "transformer_summary.json",
    )
    mlflow.end_run()


if __name__ == "__main__":
    main()
