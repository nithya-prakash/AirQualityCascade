"""Phase 7: train the spatial GNN (Model D) on PM2.5 t+1h/t+2h, evaluated
on the same chronological test period as the Phase 5/6 models.

Usage:
    python scripts/train_gnn.py
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

from aqcascade.evaluation.metrics import compute_regression_metrics  # noqa: E402
from aqcascade.graph.build_graph import build_edge_index  # noqa: E402
from aqcascade.graph.snapshots import (  # noqa: E402
    build_snapshots,
    compute_feature_medians,
    get_gnn_feature_columns,
    impute_with_medians,
)
from aqcascade.models.features import build_feature_matrix, get_feature_columns  # noqa: E402
from aqcascade.models.gnn import SpatialGNN, batch_edge_index  # noqa: E402
from aqcascade.tracking.mlflow_utils import configure_mlflow, dataset_version  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models" / "gnn"

TARGET_COLS = ["target_pm25_h1", "target_pm25_h2"]
TIME_BATCH_SIZE = 64  # hourly snapshots per training step
MAX_EPOCHS = 30
PATIENCE = 4
LEARNING_RATE = 1e-3
TRAIN_FRAC, VAL_FRAC = 0.7, 0.15
RANDOM_SEED = 42


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def time_split_mask(n_t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(n_t * TRAIN_FRAC)
    val_end = int(n_t * (TRAIN_FRAC + VAL_FRAC))
    idx = np.arange(n_t)
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


def run_split(
    model,
    features,
    targets,
    edge_index,
    indices,
    device,
    n_nodes,
    target_mean,
    target_std,
    time_batch_size,
    optimizer=None,
):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_n = 0.0, 0
    all_preds, all_targets, all_mask = [], [], []

    order = np.random.permutation(indices) if is_train else indices
    for start in range(0, len(order), time_batch_size):
        batch_t = order[start : start + time_batch_size]
        b = len(batch_t)
        x = torch.from_numpy(features[batch_t]).reshape(b * n_nodes, -1).to(device)
        y = torch.from_numpy(targets[batch_t]).to(device)  # (b, n_nodes, n_horizons)
        mask = ~torch.isnan(y)
        y_filled = torch.nan_to_num(y, nan=0.0)
        y_norm = (y_filled - target_mean) / target_std

        edge_index_b = batch_edge_index(edge_index, n_nodes, b).to(device)
        with torch.set_grad_enabled(is_train):
            pred = model(x, edge_index_b).reshape(b, n_nodes, -1)
            loss = ((pred - y_norm) ** 2 * mask).sum() / mask.sum().clamp(min=1)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * mask.sum().item()
        total_n += mask.sum().item()
        if not is_train:
            all_preds.append((pred * target_std + target_mean).detach().cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_mask.append(mask.cpu().numpy())

    avg_loss = total_loss / max(total_n, 1)
    if is_train:
        return avg_loss, None, None, None
    return (
        avg_loss,
        np.concatenate(all_preds),
        np.concatenate(all_targets),
        np.concatenate(all_mask),
    )


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    configure_mlflow("aqcascade-gnn")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = get_device()
    logger.info("Using device: %s", device)

    logger.info("Loading features + neighbor graph...")
    features_path = PROCESSED_DIR / "features.parquet"
    features_df = pd.read_parquet(features_path)
    ds_version = dataset_version(features_path)
    neighbor_graph = pd.read_parquet(PROCESSED_DIR / "neighbor_graph.parquet")

    station_order = sorted(features_df["station_id"].unique())
    n_nodes = len(station_order)
    edge_index = build_edge_index(neighbor_graph, station_order)
    logger.info("%d stations, %d directed edges (symmetrized)", n_nodes, edge_index.shape[1])

    all_feature_cols = get_feature_columns(features_df)
    raw_gnn_cols = get_gnn_feature_columns(all_feature_cols)
    # Reuse the same numeric-matrix logic as the Phase 5 baselines (bool/
    # object -> float, "season" one-hot encoded with a fixed category set)
    # rather than re-deriving dtype handling here -- that's exactly the
    # class of bug a second implementation would risk repeating (an
    # earlier version of this script used a naive `dtype != object` check,
    # which pandas 3's Arrow-backed string dtype silently defeated).
    numeric_features = build_feature_matrix(features_df, raw_gnn_cols)
    gnn_feature_cols = numeric_features.columns.tolist()
    numeric_features["station_id"] = features_df["station_id"].to_numpy()
    numeric_features["timestamp"] = features_df["timestamp"].to_numpy()
    logger.info(
        "%d node feature columns (excludes neighbor_* summaries -- see module docstring)",
        len(gnn_feature_cols),
    )

    timestamps, feature_array, target_array = build_snapshots(
        numeric_features.assign(**{c: features_df[c].to_numpy() for c in TARGET_COLS}),
        station_order,
        gnn_feature_cols,
        TARGET_COLS,
    )
    logger.info("%d hourly snapshots, feature array shape %s", len(timestamps), feature_array.shape)

    train_idx, val_idx, test_idx = time_split_mask(len(timestamps))
    logger.info(
        "Split: train=%d (%s -> %s) val=%d (%s -> %s) test=%d (%s -> %s)",
        len(train_idx),
        timestamps[train_idx[0]],
        timestamps[train_idx[-1]],
        len(val_idx),
        timestamps[val_idx[0]],
        timestamps[val_idx[-1]],
        len(test_idx),
        timestamps[test_idx[0]],
        timestamps[test_idx[-1]],
    )

    train_mask_t = np.zeros(len(timestamps), dtype=bool)
    train_mask_t[train_idx] = True
    medians = compute_feature_medians(feature_array, train_mask_t)
    feature_array = impute_with_medians(feature_array, medians)

    target_mean_np = np.nanmean(target_array[train_idx], axis=(0, 1))
    target_std_np = np.nanstd(target_array[train_idx], axis=(0, 1))
    target_mean = torch.tensor(target_mean_np, dtype=torch.float32, device=device)
    target_std = torch.tensor(target_std_np, dtype=torch.float32, device=device)
    logger.info("Target normalization (train-only): mean=%s std=%s", target_mean_np, target_std_np)

    model = SpatialGNN(n_features=len(gnn_feature_cols), n_horizons=len(TARGET_COLS)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", n_params)

    mlflow.start_run(run_name="gnn")
    mlflow.log_params(
        {
            "model_type": "gnn",
            "targets": ",".join(TARGET_COLS),
            "n_stations": n_nodes,
            "n_edges": int(edge_index.shape[1]),
            "time_batch_size": TIME_BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "random_seed": RANDOM_SEED,
            "n_features": len(gnn_feature_cols),
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
        train_loss, *_ = run_split(
            model,
            feature_array,
            target_array,
            edge_index,
            train_idx,
            device,
            n_nodes,
            target_mean,
            target_std,
            TIME_BATCH_SIZE,
            optimizer,
        )
        val_loss, *_ = run_split(
            model,
            feature_array,
            target_array,
            edge_index,
            val_idx,
            device,
            n_nodes,
            target_mean,
            target_std,
            TIME_BATCH_SIZE,
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
    for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        _, preds, targets, mask = run_split(
            model,
            feature_array,
            target_array,
            edge_index,
            idx,
            device,
            n_nodes,
            target_mean,
            target_std,
            TIME_BATCH_SIZE,
        )
        for i, target_col in enumerate(TARGET_COLS):
            m = mask[:, :, i]
            metrics = compute_regression_metrics(targets[:, :, i][m], preds[:, :, i][m])
            results.append({"target": target_col, "model": "gnn", "split": split_name, **metrics})

    results_df = pd.DataFrame(results)
    results_df.to_csv(PROCESSED_DIR / "gnn_results.csv", index=False)
    logger.info(
        "\n%s",
        results_df[results_df.split == "test"][["target", "mae", "rmse", "r2", "n"]].to_string(
            index=False
        ),
    )

    summary = {
        "device": str(device),
        "n_stations": n_nodes,
        "n_edges": int(edge_index.shape[1]),
        "n_features": len(gnn_feature_cols),
        "feature_columns": gnn_feature_cols,
        "n_parameters": n_params,
        "epochs_trained": len(history),
        "total_train_seconds": round(total_train_s, 1),
        "split_sizes_hours": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "training_history": history,
        "test_metrics": results_df[results_df.split == "test"].to_dict(orient="records"),
    }
    (PROCESSED_DIR / "gnn_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "Wrote %s and %s", PROCESSED_DIR / "gnn_results.csv", PROCESSED_DIR / "gnn_summary.json"
    )

    mlflow.log_metric("total_train_seconds", total_train_s)
    mlflow.log_metric("epochs_trained", len(history))
    for _, row in results_df.iterrows():
        for metric_name in ("mae", "rmse", "r2"):
            mlflow.log_metric(f"{row['split']}_{row['target']}_{metric_name}", row[metric_name])
    mlflow.log_artifact(str(MODELS_DIR / "best_model.pt"))
    mlflow.end_run()


if __name__ == "__main__":
    main()
