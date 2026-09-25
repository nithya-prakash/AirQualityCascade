import torch

from aqcascade.models.gnn import SpatialGNN, batch_edge_index


def test_forward_pass_shape():
    model = SpatialGNN(n_features=10, n_horizons=2, hidden_dim=16, num_layers=2)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    x = torch.randn(3, 10)
    out = model(x, edge_index)
    assert out.shape == (3, 2)


def test_batch_edge_index_offsets_each_copy_correctly():
    edge_index = torch.tensor([[0, 1], [1, 0]])  # a 2-node graph, one bidirectional edge
    batched = batch_edge_index(edge_index, n_nodes=2, batch_size=3)
    # 3 disconnected copies: nodes (0,1), (2,3), (4,5)
    assert batched.shape == (2, 6)
    edges = set(zip(batched[0].tolist(), batched[1].tolist(), strict=True))
    assert edges == {(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4)}


def test_batched_forward_matches_looped_single_graph_forward():
    """A batched forward pass over B copies of the same graph must give the
    identical per-copy output as running each copy through the model one at
    a time -- proves the block-diagonal batching doesn't leak information
    between the B independent snapshots."""
    torch.manual_seed(0)
    model = SpatialGNN(n_features=4, n_horizons=1, hidden_dim=8, num_layers=2)
    model.eval()

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    n_nodes = 3
    batch_size = 2
    x_per_snapshot = torch.randn(batch_size, n_nodes, 4)

    with torch.no_grad():
        looped = torch.stack([model(x_per_snapshot[i], edge_index) for i in range(batch_size)])

        x_batched = x_per_snapshot.reshape(batch_size * n_nodes, 4)
        edge_index_batched = batch_edge_index(edge_index, n_nodes, batch_size)
        out_batched = model(x_batched, edge_index_batched).reshape(batch_size, n_nodes, 1)

    assert torch.allclose(looped, out_batched, atol=1e-5)
