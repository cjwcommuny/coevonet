from typing import List

from arraycontract import shape
from torch import Tensor, nn
import torch

from gcn_ranker.modules.graph.gcn import GraphConvolutionLayer
from gcn_ranker.modules.graph.graph import bidirectional_graph




class GcnManifoldNaive(nn.Module):
    def __init__(self, gcn_dims: List[int]):
        super().__init__()
        self.gcn_dims = gcn_dims
        self.gcns = nn.ModuleList([
            GraphConvolutionLayer(in_dim, out_dim) for in_dim, out_dim in zip(gcn_dims[:-1], gcn_dims[1:])
        ])
        self.fc = nn.Linear(gcn_dims[0], gcn_dims[-1])
        self.scorer = nn.Linear(gcn_dims[-1], 1)

    @shape(('N', 'd'))
    def pooling(self, tensor: Tensor):
        return torch.mean(tensor, dim=0)

    @shape(frames=('N', 'd_feature'))
    def forward(self, frames: Tensor, segment_indices: List[Tensor]):
        N, d_feature = frames.shape
        adj = bidirectional_graph(N).to(frames.device)
        frames_original = frames
        for layer in self.gcns:
            frames = torch.relu(layer(frames, adj))
        frames += self.fc(frames_original)
        #
        features = torch.stack([
            self.pooling(frames[indices].reshape(-1, self.gcn_dims[-1]))
            for indices in segment_indices
        ])
        scores = self.scorer(features).reshape(-1)
        return scores
