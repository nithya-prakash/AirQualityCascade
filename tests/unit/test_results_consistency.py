"""README headline numbers and docs/details.md tables must agree with the committed result CSV."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
README = (ROOT / "README.md").read_text()
DETAILS = (ROOT / "docs" / "details.md").read_text()
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
    assert "83,970" in README and "83,970" in DETAILS
    assert {r["target"] for r in ROWS} == {"target_pm25_h1", "target_pm25_h2"}


def test_comparison_table_matches_the_csv() -> None:
    seen = set()
    for line in DETAILS.splitlines():
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
    for line in DETAILS.splitlines():
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


def test_readme_headline_numbers_match_the_csv() -> None:
    def r2(model: str, horizon: str) -> str:
        return f"{float(BY[(model, horizon)]['r2']):.3f}"

    expected = [
        f"Transformer {r2('transformer', 'h1')}",
        f"XGBoost (served) {r2('xgboost', 'h1')}",
        f"persistence {r2('persistence', 'h1')}",
        f"XGBoost {r2('xgboost', 'h2')}",
        f"GNN {r2('gnn', 'h2')}",
        f"Transformer {r2('transformer', 'h2')}",
        f"persistence {r2('persistence', 'h2')}",
        f"R² {r2('ablation_B', 'h1')} to {r2('ablation_C', 'h1')}",
        f"{r2('ablation_B', 'h2')} to {r2('ablation_C', 'h2')}",
        f"tier C {r2('ablation_C', 'h1')} vs GNN {r2('gnn', 'h1')}",
        f"{r2('ablation_C', 'h2')} vs {r2('gnn', 'h2')}",
    ]
    for text in expected:
        assert text in README, text
