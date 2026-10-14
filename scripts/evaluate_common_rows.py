"""Re-score every comparable PM2.5 model on one common set of test rows.

    python scripts/evaluate_common_rows.py

The per-model scripts score different rows (see aqcascade.evaluation.common_rows), so their
metrics cannot be ranked against each other. This script:

1. regenerates each model's predictions for every test row it can score (saved baselines and the
   Transformer/GNN checkpoints are loaded; ablation tiers A/B/C are retrained with the ablation
   script's exact procedure, since those runs did not save their models);
2. checks that each model reproduces its previously reported metric on its *own* original rows
   (the gate: if this fails, the regenerated predictions are not trustworthy);
3. scores all models on the common rows: test rows where both PM2.5 targets and the current
   PM2.5 value are present.

Writes results/common_rows_results.csv and results/common_rows_gate.csv. Nothing here trains the
Transformer or GNN.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from aqcascade.evaluation.common_rows import (  # noqa: E402
    score_models,
    select_common_rows,
    series_by_key,
)
from aqcascade.evaluation.metrics import compute_regression_metrics  # noqa: E402
from aqcascade.evaluation.split import chronological_split  # noqa: E402
from aqcascade.graph.build_graph import build_edge_index  # noqa: E402
from aqcascade.graph.snapshots import (  # noqa: E402
    build_snapshots,
    compute_feature_medians,
    get_gnn_feature_columns,
    impute_with_medians,
)
from aqcascade.models.baselines import make_xgboost  # noqa: E402
from aqcascade.models.feature_groups import (  # noqa: E402
    MODEL_A_COLUMNS,
    MODEL_B_COLUMNS,
    MODEL_C_COLUMNS,
)
from aqcascade.models.features import build_feature_matrix, get_feature_columns  # noqa: E402
from aqcascade.models.gnn import SpatialGNN, batch_edge_index  # noqa: E402
from aqcascade.models.sequence_data import (  # noqa: E402
    SEQUENCE_FEATURE_COLUMNS,
    PollutionSequenceDataset,
    apply_normalization,
    build_samples,
    build_station_arrays,
    compute_normalization_stats,
)
from aqcascade.models.transformer import PollutionTransformer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

TARGETS = ["target_pm25_h1", "target_pm25_h2"]
CURRENT = "pm25"
TIERS = {"A": MODEL_A_COLUMNS, "B": MODEL_B_COLUMNS, "C": MODEL_C_COLUMNS}
SEQ_LEN = 24
# Saved artifacts must reproduce almost exactly. The ablation tiers are retrained here (their models
# were never saved), so some retraining noise on a different platform is expected and allowed.
GATE_TOLERANCE_R2 = {"saved": 0.001, "retrained": 0.01}
Preds = dict[str, dict[str, pd.Series]]  # model -> target -> predictions indexed by key


def tabular_predictions(features: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> Preds:
    cols = get_feature_columns(features)
    x_test = build_feature_matrix(test, cols)
    out: Preds = {"persistence": {}}
    for target in TARGETS:
        h = target.rsplit("_", 1)[1]
        out["persistence"][target] = series_by_key(test, test[CURRENT])
        for name in ("linear_regression", "random_forest", "xgboost"):
            model = joblib.load(MODELS / "baselines" / f"target_pm25_{h}__{name}.joblib")
            out.setdefault(name, {})[target] = series_by_key(test, model.predict(x_test))
        # Ablation tiers: same procedure as scripts/run_ablation.py (target-present training rows).
        train_t = train.dropna(subset=[target])
        for tier, tier_cols in TIERS.items():
            model = make_xgboost()
            model.fit(build_feature_matrix(train_t, tier_cols), train_t[target])
            preds = model.predict(build_feature_matrix(test, tier_cols))
            out.setdefault(f"ablation_{tier}", {})[target] = series_by_key(test, preds)
    return out


def transformer_predictions(features: pd.DataFrame) -> dict[str, pd.Series]:
    arrays, timestamps, channels = build_station_arrays(features)
    samples = build_samples(features, TARGETS, SEQ_LEN)
    train_s, _, test_s = chronological_split(samples, "timestamp", 0.7, 0.15)
    skip = {"hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos"}
    skip |= {"wind_direction_sin", "wind_direction_cos"}
    normalize = [c for c in SEQUENCE_FEATURE_COLUMNS if c not in skip]
    stats = compute_normalization_stats(
        arrays, timestamps, channels, normalize, train_s["timestamp"].max()
    )
    norm = apply_normalization(arrays, channels, normalize, stats)
    mean = train_s[TARGETS].to_numpy().mean(axis=0).astype("float32")
    std = train_s[TARGETS].to_numpy().std(axis=0).astype("float32")

    model = PollutionTransformer(n_channels=len(channels), n_horizons=len(TARGETS), seq_len=SEQ_LEN)
    model.load_state_dict(
        torch.load(MODELS / "transformer" / "best_model.pt", weights_only=True, map_location="cpu")
    )
    model.eval()
    loader = DataLoader(
        PollutionSequenceDataset(test_s, norm, TARGETS, SEQ_LEN), batch_size=1024, shuffle=False
    )
    with torch.no_grad():
        preds = np.concatenate([model(x).numpy() * std + mean for x, _ in loader])
    return {t: series_by_key(test_s, preds[:, i]) for i, t in enumerate(TARGETS)}


def gnn_predictions(features: pd.DataFrame) -> dict[str, pd.Series]:
    graph = pd.read_parquet(PROCESSED / "neighbor_graph.parquet")
    stations = sorted(features["station_id"].unique())
    n_nodes = len(stations)
    edge_index = build_edge_index(graph, stations)
    numeric = build_feature_matrix(features, get_gnn_feature_columns(get_feature_columns(features)))
    gnn_cols = numeric.columns.tolist()
    numeric["station_id"] = features["station_id"].to_numpy()
    numeric["timestamp"] = features["timestamp"].to_numpy()
    stamps, feats, targets = build_snapshots(
        numeric.assign(**{c: features[c].to_numpy() for c in TARGETS}), stations, gnn_cols, TARGETS
    )
    n_t = len(stamps)
    train_idx = np.arange(int(n_t * 0.7))
    test_idx = np.arange(int(n_t * 0.85), n_t)
    train_mask = np.zeros(n_t, dtype=bool)
    train_mask[train_idx] = True
    feats = impute_with_medians(feats, compute_feature_medians(feats, train_mask))
    mean = np.nanmean(targets[train_idx], axis=(0, 1))
    std = np.nanstd(targets[train_idx], axis=(0, 1))

    model = SpatialGNN(n_features=len(gnn_cols), n_horizons=len(TARGETS))
    model.load_state_dict(
        torch.load(MODELS / "gnn" / "best_model.pt", weights_only=True, map_location="cpu")
    )
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(test_idx), 64):
            batch = test_idx[start : start + 64]
            b = len(batch)
            x = torch.from_numpy(feats[batch]).reshape(b * n_nodes, -1)
            pred = model(x, batch_edge_index(edge_index, n_nodes, b)).reshape(b, n_nodes, -1)
            chunks.append(pred.numpy() * std + mean)
    preds = np.concatenate(chunks)  # (n_test_hours, n_nodes, n_horizons)
    keys = pd.DataFrame(
        {
            "station_id": np.tile(np.asarray(stations), len(test_idx)),
            "timestamp": np.repeat(pd.to_datetime(stamps[test_idx]), n_nodes),
        }
    )
    keys["timestamp"] = keys["timestamp"].astype(features["timestamp"].dtype)
    return {t: series_by_key(keys, preds[:, :, i].reshape(-1)) for i, t in enumerate(TARGETS)}


def gate(
    preds: Preds, features: pd.DataFrame, test: pd.DataFrame, saved: dict[tuple[str, str], float]
) -> pd.DataFrame:
    """Each model's reproduced R2 on its own original rows vs the R2 reported earlier.

    Truth comes from the full table: the Transformer's own test rows start earlier than the shared
    test window (its split is computed on valid samples only), so they are not all inside `test`.
    """
    truth_all = features.set_index(["station_id", "timestamp"])
    records = []
    for model, by_target in preds.items():
        for target, series in by_target.items():
            truth = truth_all[target]
            if model in ("persistence", "linear_regression", "random_forest", "xgboost"):
                rows = test.dropna(subset=[target, CURRENT])  # baseline script's rows
                index = pd.MultiIndex.from_frame(rows[["station_id", "timestamp"]])
            elif model.startswith("ablation_") or model == "gnn":
                rows = test.dropna(subset=[target])  # target-present rows
                index = pd.MultiIndex.from_frame(rows[["station_id", "timestamp"]])
            else:  # transformer: its own samples (full window, both targets) are the series index
                index = series.index
            y, pred = truth.reindex(index), series.reindex(index)
            if pred.isna().any() or y.isna().any():
                raise ValueError(f"{model}/{target}: predictions missing on its own original rows")
            r2 = compute_regression_metrics(y.to_numpy(), pred.to_numpy())["r2"]
            ref = saved.get((model, target))
            kind = "retrained" if model.startswith("ablation_") else "saved"
            records.append(
                {
                    "model": model,
                    "target": target,
                    "kind": kind,
                    "own_rows_n": len(index),
                    "reported_r2": ref,
                    "reproduced_r2": r2,
                    "delta": None if ref is None else r2 - ref,
                }
            )
    return pd.DataFrame(records)


def saved_reference() -> dict[tuple[str, str], float]:
    ref: dict[tuple[str, str], float] = {}
    for fname, name_of in (
        ("baseline_results.csv", lambda r: r["model"]),
        ("ablation_results.csv", lambda r: f"ablation_{r['tier']}"),
        ("transformer_results.csv", lambda r: "transformer"),
        ("gnn_results.csv", lambda r: "gnn"),
    ):
        df = pd.read_csv(PROCESSED / fname)
        for _, r in df[df["split"] == "test"].iterrows():
            ref[(name_of(r), r["target"])] = float(r["r2"])
    return ref


def main() -> int:
    features = pd.read_parquet(PROCESSED / "features.parquet")
    train, _, test = chronological_split(features, "timestamp", 0.7, 0.15)
    keys = select_common_rows(test, TARGETS, CURRENT)
    logger.info(
        "test rows=%d, with both targets=%d, common rows (also current pm25)=%d",
        len(test),
        len(test.dropna(subset=TARGETS)),
        len(keys),
    )

    preds = tabular_predictions(features, train, test)
    preds["transformer"] = transformer_predictions(features)
    preds["gnn"] = gnn_predictions(features)

    gate_df = gate(preds, features, test, saved_reference())
    RESULTS.mkdir(exist_ok=True)
    gate_df.to_csv(RESULTS / "common_rows_gate.csv", index=False)
    print("\n== Reproduction gate (own original rows) ==")
    print(gate_df.round(4).to_string(index=False))
    tolerance = gate_df["kind"].map(GATE_TOLERANCE_R2)
    failed = gate_df[gate_df["delta"].abs() > tolerance]
    print(f"\nGate tolerance on R2: {GATE_TOLERANCE_R2}; failures: {len(failed)}")

    truth_frame = test.set_index(["station_id", "timestamp"])
    frames = []
    for target in TARGETS:
        truth = truth_frame[target]
        scored = score_models(keys, truth, {m: p[target] for m, p in preds.items()})
        scored.insert(0, "target", target)
        frames.append(scored)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(RESULTS / "common_rows_results.csv", index=False)
    print("\n== Common rows ==")
    print(out.round(4).to_string(index=False))
    summary = {"test_rows": len(test), "common_rows": len(keys), "gate_failures": len(failed)}
    print("\n" + json.dumps(summary))
    return 1 if len(failed) else 0


if __name__ == "__main__":
    sys.exit(main())
