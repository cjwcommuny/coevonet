from functools import partial
from typing import List, Tuple

import torch
from arraycontract import shape
from torch import nn, Tensor

from gcn_ranker.modules.graph.gcn import MultiLayerGcn
from gcn_ranker.modules.graph.gcn_clustering import generate_weight_adj
from gcn_ranker.modules.graph.graph import bidirectional_graph
from gcn_ranker.modules.graph.hetero_graph import MultiLayerHeteroGnn
from gcn_ranker.modules.nlp.text2embedding import TextProcessor


class MultiModalGnn(nn.Module):
    def __init__(
            self,
            video_bi_dims: List[int],
            video_weight_dims: List[int],
            video_weight_normalize: str,
            video_weight_project: str,
            hetero_node_dims: List[Tuple[int]],
            hetero_edge_dims: List[int],
            text_processor_model: str,
            dropout: float,
            pooling: str,
    ):
        super().__init__()
        self.video_gcn_dims = video_bi_dims
        self.adj_normalized = {
            'mean': torch.mean,
            'max': torch.max,
            'min': torch.min,
            'median': torch.median
        }[video_weight_normalize]
        self.adj_weight_project = {
            'negative_exp': lambda x: torch.exp(-x)
        }[video_weight_project]
        self.pooling = {
            'mean': lambda x: torch.mean(x, dim=0),
            'max': lambda x: torch.max(x, dim=0).values
        }[pooling]
        #
        self.video_bi_gcn = MultiLayerGcn(video_bi_dims, dropout)
        self.video_weight_gcn = MultiLayerGcn(video_weight_dims, dropout)
        self.text_processor = TextProcessor(text_processor_model)
        self.rnn = nn.GRU(self.text_processor.output_dim, hetero_node_dims[0][0])
        self.hetero_gnn = MultiLayerHeteroGnn(hetero_node_dims, hetero_edge_dims, nn.ReLU())
        self.fc = nn.Linear(hetero_node_dims[-1][1], 1)

    def forward(self, *args, **kwargs):
        return self.debug_forward(*args, **kwargs)[0]

    @shape(frame_features=('num_frames', 'd'))
    def debug_forward(self, text: str, frame_features: Tensor, segment_indices: List[Tensor]):
        num_frames = frame_features.shape[0]
        bi_frame_features = self.video_bi_gcn(
            frame_features,
            bidirectional_graph(num_frames).to(frame_features.device)
        )
        weight_adj = generate_weight_adj(
            bi_frame_features,
            normlize=self.adj_normalized,
            project=self.adj_weight_project
        )
        weight_frame_features = self.video_weight_gcn(bi_frame_features, weight_adj)
        text_feature = self.text_processor(text)
        assert text_feature.ndim == 2
        text_feature = self.rnn(text_feature.unsqueeze(dim=1))[1].reshape(-1)
        text_feature_hetero, frame_features_hetero = self.hetero_gnn(text_feature, weight_frame_features)
        segment_features = torch.stack(
            [self.pooling(frame_features_hetero[indices]) for indices in segment_indices]
        )
        segment_scores = self.fc(segment_features).view(-1)
        return (
            segment_scores,
            bi_frame_features,
            weight_frame_features,
            weight_adj,
            text_feature,
            text_feature_hetero,
            frame_features_hetero,
            segment_features
        )
