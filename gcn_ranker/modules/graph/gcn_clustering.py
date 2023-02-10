from typing import List, Callable

import torch
from arraycontract import shape
from torch import nn, Tensor

from gcn_ranker.modules.graph.gcn import MultiLayerGcn
from gcn_ranker.modules.graph.graph import bidirectional_graph, fully_connected_graph


@shape(vectors=('N', 'd'))
def pairwise_l2_norm(vectors: Tensor):
    return torch.sqrt(torch.mul(vectors, vectors).sum(dim=1))

@shape(vectors=('N', 'd'))
def generate_weight_adj(vectors: Tensor, normlize: Callable, project: Callable) -> Tensor:
    num_nodes = vectors.shape[0]
    previous, next = vectors[:-1,:], vectors[1:,:]
    distances = pairwise_l2_norm(previous - next)
    distances = distances / normlize(distances)
    distance_accumulated = torch.cumsum(
        torch.cat([torch.tensor(0., device=distances.device).view(1), distances]),
        dim=0
    ).reshape(1, -1).expand(size=(num_nodes, num_nodes))
    distance_matrix = torch.abs(distance_accumulated.t() - distance_accumulated) / num_nodes
    weight_matrix = project(distance_matrix)
    return weight_matrix



class GcnClustering(nn.Module):
    def __init__(
            self,
            gcn1_dims: List[int],
            gcn2_dims: List[int],
            dropout: float,
            segment_score_agg: str,
            adj_normalized: str,
            adj_weight_project: str,
            boundary_padding: int,
            graph_type: str
    ):
        super().__init__()
        self.gcn1_dims = gcn1_dims
        self.gcn2_dims = gcn2_dims
        self.segment_score_agg = {'mean': torch.mean, 'max': torch.max}[segment_score_agg]
        self.adj_normalized = {
            'mean': torch.mean,
            'max': torch.max,
            'min': torch.min,
            'median': torch.median
        }[adj_normalized]
        self.adj_weight_project = {
            'negative_exp': lambda x: torch.exp(-x)
        }[adj_weight_project]
        self.graph_type = {
            'bidirectional_graph': bidirectional_graph,
            'fully_connected_graph': fully_connected_graph
        }[graph_type]
        self.boundary_padding = boundary_padding
        #
        self.gcn1 = MultiLayerGcn(gcn1_dims, dropout)
        self.gcn2 = MultiLayerGcn(gcn2_dims, dropout)
        self.fc = nn.Linear(gcn2_dims[-1] if len(gcn2_dims) != 0 else gcn1_dims[-1], 1)

    @shape(frame_features=('num_frames', 'd'))
    def forward(self, frame_features: Tensor, segment_indices: List[Tensor]):
        return self.debug_forward(frame_features, segment_indices)[0]

    def debug_forward(self, frame_features: Tensor, segment_indices: List[Tensor]):
        frame_features = self.pad(frame_features, self.boundary_padding)
        num_frames = frame_features.shape[0]
        features1 = self.gcn1(frame_features, self.graph_type(num_frames).to(frame_features.device))
        weight_adj = generate_weight_adj(
            features1,
            normlize=self.adj_normalized,
            project=self.adj_weight_project
        ).to(frame_features.device)
        features2 = self.gcn2(features1, weight_adj)
        # unpad
        weight_adj = weight_adj[self.boundary_padding:weight_adj.shape[0] - self.boundary_padding, self.boundary_padding:weight_adj.shape[0] - self.boundary_padding]
        features1 = features1[self.boundary_padding:features1.shape[0] - self.boundary_padding]
        features2 = features2[self.boundary_padding:features2.shape[0] - self.boundary_padding]
        #
        frame_scores = self.fc(features2).reshape(-1)
        segment_scores = torch.cat(
            [self.segment_score_agg(frame_scores[indices]).view(1) for indices in segment_indices]
        )
        return segment_scores, weight_adj, features1, features2

    @staticmethod
    def pad(frame_features: Tensor, pad_num: int):
        d_feature = frame_features.shape[1]
        head = frame_features[0,:].view(1, -1).expand(pad_num, d_feature)
        tail = frame_features[-1,:].view(1,-1).expand(pad_num, d_feature)
        return torch.cat([head, frame_features, tail])
