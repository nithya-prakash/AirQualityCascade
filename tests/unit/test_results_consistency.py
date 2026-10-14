"""The README comparison tables must agree with the committed result CSV."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
README = (ROOT / "README.md").read_text()
ROWS = list(csv.DictReader((ROOT / "results" / "common_rows_results.csv").open()))
BY = {(r["model"], r["target"][-2:]): r for r in ROWS}

COMPARISON = {
    "Persistence": "persistence",
    "Linear regression": "linear_regression",
    "Random forest": "random_forest",
    "XGBoost (Model C, the served model)": "xgboost",
    "GNN": "gnn",
    "Transformer (no neighbour features)": "transformer",
}
TIERS = {
    "A — local history only": "A",
    "B — A + weather": "B",
    "D — B + GNN message passing": "gnn",
}


def _cells(line: str) -> list[str]:
    return [c.strip().strip("*") for c in line.strip().strip("|").split("|")]


def test_every_model_is_scored_on_the_same_number_of_rows() -> None:
    assert {r["n"] for r in ROWS} == {"83970"}
    assert "83,970" in README
    assert {r["target"] for r in ROWS} == {"target_pm25_h1", "target_pm25_h2"}


def test_comparison_table_matches_the_csv() -> None:
    seen = set()
    for line in README.splitlines():
        cells = _cells(line)
        if line.startswith("|") and cells[0] in COMPARISON and len(cells) == 7:
            model = COMPARISON[cells[0]]
            expected = [
                f"{float(BY[(model, h)][k]):.3f}"
                for h in ("h1", "h2")
                for k in ("mae", "rmse", "r2")
            ]
            assert cells[1:] == expected, cells[0]
            seen.add(model)
    assert seen == set(COMPARISON.values())


def test_spatial_table_matches_the_csv() -> None:
    seen = set()
    for line in README.splitlines():
        cells = _cells(line)
        if (
            line.startswith("|")
            and len(cells) == 4
            and cells[0].split(" ")[0] in {"A", "B", "C", "D"}
        ):
            label = cells[0].replace("**", "")
            key = TIERS.get(label, "C" if label.startswith("C —") else None)
            assert key, label
            model = key if key == "gnn" else f"ablation_{key}"
            assert cells[2:] == [f"{float(BY[(model, h)]['r2']):.3f}" for h in ("h1", "h2")], label
            seen.add(model)
    assert seen == {"ablation_A", "ablation_B", "ablation_C", "gnn"}
