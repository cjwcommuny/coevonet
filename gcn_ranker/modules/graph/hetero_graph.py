from typing import List, Tuple

import torch
from arraycontract import shape
from torch import Tensor, nn

from gcn_ranker.modules.common import NodeAttention, GeneralSequential


class HeteroGraph(nn.Module):
    def __init__(
            self,
            in_center_dim: int,
            in_context_dim: int,
            out_center_dim: int,
            out_context_dim: int,
            edge_dim: int,
            activation: nn.Module
    ):
        super().__init__()
        self.attention = NodeAttention(in_center_dim, in_context_dim, edge_dim, activation)
        self.context_node_updator = nn.Sequential(
            nn.Linear(in_context_dim + edge_dim, out_context_dim),
            activation
        )
        self.center_node_updator = nn.Sequential(
            nn.Linear(in_center_dim + edge_dim, out_center_dim),
            activation
        )

    @shape(center=('in_center_dim',), context=('N', 'in_context_dim'))
    def forward(self, center: Tensor, context: Tensor):
        edges = self.attention(center, context)
        context = self.context_node_updator(torch.cat([context, edges], dim=1)) + context
        center = self.context_node_updator(torch.cat([center, torch.mean(edges, dim=0)]).view(1,-1)).view(-1)
        return center, context


class MultiLayerHeteroGnn(nn.Module):
    def __init__(self, node_dims: List[Tuple[int]], edge_dims: List[int], activation: nn.Module):
        super().__init__()
        assert len(edge_dims) == len(node_dims) - 1
        self.layers = GeneralSequential(*[
            HeteroGraph(
                in_center_dim,
                in_context_dim,
                out_center_dim,
                out_context_dim,
                edge_dim,
                activation
            ) for (in_center_dim, in_context_dim),
                  (out_center_dim, out_context_dim),
                  edge_dim
            in zip(node_dims[:-1], node_dims[1:], edge_dims)
        ])

    def forward(self, center: Tensor, context: Tensor):
        return self.layers(center, context)
