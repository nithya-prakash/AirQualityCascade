import pandas as pd

from aqcascade.features.spatial import add_spatial_features, compute_neighbor_graph


def _three_station_meta() -> pd.DataFrame:
    # Station 1 and 2 are close together (~11km, roughly 0.1 deg lat);
    # station 3 is far away (~1100km, roughly 10 deg lat) so it should never
    # show up as anyone's nearest neighbor within a normal max-distance cap.
    return pd.DataFrame(
        {
            "station_id": [1, 2, 3],
            "latitude": [52.5, 52.6, 42.5],
            "longitude": [13.4, 13.4, 13.4],
        }
    )


def test_neighbor_graph_finds_the_close_station_not_the_far_one():
    graph = compute_neighbor_graph(_three_station_meta(), k=2, max_distance_km=100)
    station1_neighbors = graph[graph.station_id == 1]["neighbor_station_id"].tolist()
    assert station1_neighbors == [2]  # station 3 excluded by max_distance_km


def test_neighbor_graph_respects_k():
    meta = pd.DataFrame(
        {
            "station_id": [1, 2, 3, 4],
            "latitude": [52.5, 52.51, 52.52, 52.53],
            "longitude": [13.4] * 4,
        }
    )
    graph = compute_neighbor_graph(meta, k=2, max_distance_km=1000)
    assert len(graph[graph.station_id == 1]) == 2


def test_add_spatial_features_excludes_self_and_averages_neighbors():
    meta = _three_station_meta()
    graph = compute_neighbor_graph(meta, k=2, max_distance_km=100)  # only station 1<->2 edge

    hours = pd.date_range("2026-01-01", periods=2, freq="h")
    df = pd.DataFrame(
        {
            "station_id": [1, 1, 2, 2],
            "timestamp": list(hours) * 2,
            "pm25": [10.0, 12.0, 20.0, 24.0],
        }
    )
    out = add_spatial_features(df, graph, "pm25")

    station1 = out[out.station_id == 1].sort_values("timestamp")
    # Station 1's only neighbor is station 2, so its neighbor mean should be
    # exactly station 2's own value, never including station 1 itself.
    assert station1["neighbor_pm25_mean"].tolist() == [20.0, 24.0]
    assert station1["neighbor_pm25_count"].tolist() == [1, 1]
