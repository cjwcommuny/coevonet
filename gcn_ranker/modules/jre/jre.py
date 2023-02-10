from typing import List, Callable

import torch
from arraycontract import shape
from torch import nn, Tensor
from torchtools.modules import Identical

from gcn_ranker.modules.common import Loop
from gcn_ranker.modules.graph.gcn import GraphConvolutionLayer
from gcn_ranker.modules.graph.graph import fully_connected_graph
import torch.nn.functional as F


class PairInteraction(nn.Module):
    def __init__(self, in_dim1: int, in_dim2: int, out_dim: int, forward_dim: int, padding_mode: str, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(in_dim1, forward_dim, bias=True)
        self.fc2 = nn.Linear(in_dim2, forward_dim, bias=False)
        self.interaction = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(forward_dim, out_dim)
        )
        #
        self.in_dim1 = in_dim1
        self.in_dim2 = in_dim2
        self.forward_dim = forward_dim
        self.out_dim = out_dim

    @shape(features1=('N1', 'd1'), features2=('N2', 'd2'))
    def forward(self, features1: Tensor, features2: Tensor) -> Tensor:
        """
        :param features1:
        :param features2:
        :return: shape=(N1, N2, out_dim)
        """
        N1, N2 = features1.shape[0], features2.shape[0]
        features1 = self.fc1(features1).view(N1, 1, self.forward_dim)
        features2 = self.fc2(features2).view(1, N2, self.forward_dim)
        features1 = features1.expand(N1, N2, self.forward_dim)
        features2 = features2.expand(N1, N2, self.forward_dim)
        return self.interaction(features1 + features2)


class JreCell(nn.Module):
    def __init__(
            self,
            video_dim: int,
            gcn_dims: List[int],
            interaction_dim: int,
            pooling: Callable,
            padding_mode: str,
            dropout: float,
            update_rate: float,
            text_encoder_type: str,
            use_interaction_module: bool
    ):
        super().__init__()
        self.video_encoder = VideoEncoder(video_dim, 2 * video_dim, video_dim, dropout)
        self.text_encoder = {
            'TextGnnEncoder': TextGnnEncoder,
            'TextGcnEncoder': TextGcnEncoder,
            'identity': Identical
        }[text_encoder_type](gcn_dims[0], gcn_dims[-1])
        self.interact_pairs = PairInteraction(
            in_dim1=video_dim,
            in_dim2=gcn_dims[-1],
            out_dim=interaction_dim,
            forward_dim=interaction_dim * 2,
            padding_mode=padding_mode,
            dropout=dropout
        ) if use_interaction_module else None
        self.text_amplifier_generator = nn.Sequential(
            nn.Linear(interaction_dim, 1),
            nn.Softmax(dim=1)
        )
        self.video_amplifier_generator = nn.Sequential(
            nn.Linear(interaction_dim, 1),
            nn.Softmax(dim=0)
        )
        self.pooling = pooling
        #
        self.interaction_dim = interaction_dim
        self.update_rate = update_rate

    def forward(
            self,
            video_features: Tensor,
            text_features: Tensor,
            video_position_embedding: Tensor,
            text_position_embedding: Tensor
    ):
        T, N = video_features.shape[0], text_features.shape[0]
        input_video_features, input_text_features = video_features, text_features
        video_features = video_features + video_position_embedding
        text_features = text_features + text_position_embedding
        video_features = self.video_encoder(video_features)
        text_features = self.text_encoder(text_features)
        if self.interact_pairs is not None:
            interaction_map = self.interact_pairs(video_features, text_features)
            #
            text_attentions = self.text_amplifier_generator(interaction_map)
            text_context = torch.sum(text_attentions * text_features.reshape(1, N, -1), dim=1)
            #
            video_attentions = self.video_amplifier_generator(interaction_map)
            video_context = torch.sum(video_attentions * video_features.reshape(T, 1, -1), dim=0)
            assert text_context.shape == (T, self.interaction_dim) and video_context.shape == (N, self.interaction_dim)
            text_features = video_context * self.update_rate + text_features * (1 - self.update_rate) + input_text_features
            video_features = text_context * self.update_rate + video_features * (1 - self.update_rate) + input_video_features
        return video_features, text_features


class Jre(nn.Module):
    def __init__(
            self,
            gcn_dims: List[int],
            pooling: Callable,
            dropout: float,
            loop_n_times: int,
            padding_mode: str,
            share: bool,
            update_rate: float,
            text_encoder_type: str,
            use_interaction_module: bool,
            fix_language_input: bool
    ):
        super().__init__()
        assert gcn_dims[0] == gcn_dims[-1]
        if share:
            jre = JreCell(
                video_dim=gcn_dims[-1],
                gcn_dims=gcn_dims,
                interaction_dim=gcn_dims[-1],
                pooling=pooling,
                padding_mode=padding_mode,
                dropout=dropout,
                update_rate=update_rate,
                text_encoder_type=text_encoder_type,
                use_interaction_module=use_interaction_module
            )
            self.loop = nn.ModuleList([jre] * loop_n_times)
        else:
            self.loop = nn.ModuleList([JreCell(
                video_dim=gcn_dims[-1],
                gcn_dims=gcn_dims,
                interaction_dim=gcn_dims[-1],
                pooling=pooling,
                padding_mode=padding_mode,
                dropout=dropout,
                update_rate=update_rate,
                text_encoder_type=text_encoder_type,
                use_interaction_module=use_interaction_module
            ) for _ in range(loop_n_times)])
        self.fix_language_input = fix_language_input

    def forward(self, video, text, video_position_embedding, text_position_embedding):
        if self.fix_language_input:
            for layer in self.loop:
                video, _ = layer(video, text, video_position_embedding, text_position_embedding)
        else:
            for layer in self.loop:
                video, text = layer(video, text, video_position_embedding, text_position_embedding)
        return video, text



class VideoEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, padding=1),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )

    @shape(video=('T', 'd'))
    def forward(self, video: Tensor):
        video = video.t().unsqueeze(0)
        video = self.layers(video).squeeze(0).t()
        return video

class TextGnnEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.bilinear = nn.Linear(in_dim, in_dim, bias=False)
        self.linear = nn.Linear(in_dim, out_dim)

    @shape(text=('N', 'd'))
    def forward(self, text: Tensor):
        N = text.shape[0]
        text = F.normalize(text, p=2, dim=1)
        edges = torch.mm(self.bilinear(text), text.t())
        assert edges.shape == (N, N), f'{edges.shape=}, {text.shape=}'
        weights = F.sigmoid(torch.mean(edges, dim=1, keepdim=True)) # shape=(N,1)
        text = weights * self.linear(text)
        return text

class TextGcnEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.gcn = GraphConvolutionLayer(in_dim, out_dim)

    def forward(self, text: Tensor):
        N = text.shape[0]
        adj = fully_connected_graph(N).to(text.device)
        return self.gcn(text, adj)
