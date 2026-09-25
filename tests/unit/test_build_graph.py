import pandas as pd
import pytest

from aqcascade.graph.build_graph import build_edge_index


def test_edges_are_symmetrized():
    # Only a directed A->B relationship exists (A's nearest neighbor is B),
    # but message passing must be able to flow B->A too.
    neighbor_graph = pd.DataFrame({"station_id": [10], "neighbor_station_id": [20]})
    edge_index = build_edge_index(neighbor_graph, station_order=[10, 20])
    edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True))
    assert (0, 1) in edges  # 10 -> 20
    assert (1, 0) in edges  # 20 -> 10 (added for symmetry)


def test_station_ids_mapped_to_positional_index():
    neighbor_graph = pd.DataFrame({"station_id": [100], "neighbor_station_id": [300]})
    edge_index = build_edge_index(neighbor_graph, station_order=[100, 200, 300])
    edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True))
    assert (0, 2) in edges  # station 100 -> index 0, station 300 -> index 2
    assert (1, 0) not in edges  # station 200 (index 1) has no edges


def test_unknown_station_raises():
    neighbor_graph = pd.DataFrame({"station_id": [999], "neighbor_station_id": [1]})
    with pytest.raises(ValueError, match="not in station_order"):
        build_edge_index(neighbor_graph, station_order=[1, 2])


def test_duplicate_edges_deduplicated():
    # A already-symmetric pair should not produce duplicate entries.
    neighbor_graph = pd.DataFrame({"station_id": [1, 2], "neighbor_station_id": [2, 1]})
    edge_index = build_edge_index(neighbor_graph, station_order=[1, 2])
    assert edge_index.shape[1] == 2  # (0,1) and (1,0), not 4
