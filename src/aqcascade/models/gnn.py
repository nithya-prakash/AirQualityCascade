"""Spatial GNN forecaster (Model D): GraphSAGE message passing over the
static k-NN station graph, applied to each hourly snapshot independently.

Design decision (documented per the brief's explicit allowance): a full
spatio-temporal architecture (sequence encoder + graph conv, as in the
brief's "potential architecture") is not implemented here. It would be
largely redundant with what's already in the node features: Phase 4's
lag/rolling/trend features already encode each station's own temporal
history, so the one thing left for the GNN to contribute is the spatial
aggregation step -- and that's what GraphSAGE does. Combining a from-scratch
temporal encoder AND a GNN in one model would need materially more data or
regularization than 90 days x 315 stations supports without a real risk of
overfitting an already-small dataset for the questionable benefit of
re-deriving temporal features an engineered pipeline already provides more
directly. This is the "simplest scientifically defensible graph
architecture" the brief allows for in that situation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class SpatialGNN(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_horizons: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden_dim)
        self.convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_horizons),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.input_proj(x))
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
            h = self.dropout(h)
        return self.head(h)


def batch_edge_index(edge_index: torch.Tensor, n_nodes: int, batch_size: int) -> torch.Tensor:
    """Block-diagonal edge_index for `batch_size` disconnected copies of the
    same static graph -- lets one forward pass cover several hourly
    snapshots at once (each snapshot's nodes never connect to another
    snapshot's), which is far faster than one tiny 315-node graph per step.
    """
    device = edge_index.device
    offsets = (torch.arange(batch_size, device=device) * n_nodes).view(-1, 1, 1)
    expanded = edge_index.unsqueeze(0) + offsets  # (batch_size, 2, E)
    return expanded.permute(1, 0, 2).reshape(2, -1)
