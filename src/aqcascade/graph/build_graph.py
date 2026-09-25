"""Turns the Phase 4 k-nearest-neighbor station graph into a PyG edge_index.

Reuses `data/processed/neighbor_graph.parquet` exactly as built in
src/aqcascade/features/spatial.py -- the same graph the Phase 4/5 "neighbor
mean" features were computed from, so Model C (hand-aggregated neighbor
features) and Model D (this GNN, which learns the aggregation itself) are
compared on identical spatial structure, not two different notions of
"neighbor".
"""

from __future__ import annotations

import pandas as pd
import torch


def build_edge_index(neighbor_graph: pd.DataFrame, station_order: list[int]) -> torch.Tensor:
    """Directed edge_index (2, E) in node-index space (0..n_stations-1),
    made symmetric: a k-NN relationship is directional ("A's nearest
    neighbor is B" doesn't imply "B's nearest neighbor is A"), but message
    passing should let information flow both ways -- air quality
    correlation between two nearby points isn't one-directional just
    because a distance ranking happens to be.
    """
    station_idx = {sid: i for i, sid in enumerate(station_order)}

    src = neighbor_graph["station_id"].map(station_idx)
    dst = neighbor_graph["neighbor_station_id"].map(station_idx)
    if src.isna().any() or dst.isna().any():
        raise ValueError("neighbor_graph references a station_id not in station_order")

    edges = set(zip(src.astype(int), dst.astype(int), strict=True))
    edges |= {(b, a) for a, b in edges}  # symmetrize

    sorted_edges = sorted(edges)
    src_list = [e[0] for e in sorted_edges]
    dst_list = [e[1] for e in sorted_edges]
    return torch.tensor([src_list, dst_list], dtype=torch.long)
