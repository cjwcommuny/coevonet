from collections import OrderedDict
from typing import Optional, Tuple

import torch
from arraycontract import shape
from torch import nn, Tensor
from torchtools.modules import AdditiveAttention
from transformers import AutoModelWithLMHead

from gcn_ranker.modules.common import View


class Lnlv(nn.Module):
    """
    Chen, Jingyuan, et al. “Localizing natural language in videos.” Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 33. 2019.
    """
    def __init__(
            self,
            vocab_size: Optional[int],
            word_embed_dim: Optional[int],
            pretrained_word_embedding: Optional[str],
            video_feature_dim: int,
            hidden_dim: int,
            dropout: float,
            window_size: int
    ):
        super().__init__()
        assert hidden_dim % 2 == 0
        self.video_pipeline = nn.Sequential(OrderedDict([
            ('video_projector', nn.Linear(video_feature_dim, hidden_dim)),
            ('dropout', nn.Dropout(dropout)),
            ('view', View(-1, 1, hidden_dim)),
            ('gru', nn.GRU(hidden_dim, hidden_dim // 2, bidirectional=True)),
        ]))
        #
        word_embedding = nn.Embedding(vocab_size, word_embed_dim) \
            if pretrained_word_embedding is None \
            else AutoModelWithLMHead.from_pretrained(pretrained_word_embedding) \
            .bert.embeddings.word_embeddings
        self.text_pipeline = nn.Sequential(OrderedDict([
            ('word_embedding', word_embedding),
            ('word_projector', nn.Linear(word_embedding.weight.shape[1], hidden_dim)),
            ('dropout', nn.Dropout(dropout)),
            ('view', View(-1, 1, hidden_dim)),
            ('gru', nn.GRU(hidden_dim, hidden_dim // 2, bidirectional=True))
        ]))
        #
        self.cross_modal_interactor = CrossModalInteractor(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
        self.self_interactor = SelfInteractor(hidden_dim)
        self.segment_localizer = SegmentLocalizer(hidden_dim, hidden_dim, hidden_dim, dropout)
        #
        self.window_size = window_size

    @shape(video=('T', 'd'), text=('N',))
    def forward(self, video: Tensor, text: Tensor):
        """
        :return
            - window_scores: shape=(num_window,)
            - window_starts: shape=(num_window,)
        """
        T, S = video.shape[0], text.shape[0]
        H_v = self.video_pipeline(video)[0].view(T, -1)
        H_s = self.text_pipeline(text)[0].view(S, -1)
        #
        h_r = self.cross_modal_interactor(H_v, H_s)
        h_d_forward = self.self_interactor(h_r)
        frame_scores = self.segment_localizer(h_d_forward, H_s)
        #
        window_starts = torch.arange(start=0, end=T - self.window_size + 1, step=1).to(video.device)
        window_scores = frame_scores.index_select(dim=0, index=window_starts).view(-1)
        return window_scores, window_starts


class CrossModalInteractor(nn.Module):
    def __init__(self, video_dim: int, text_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.attention = AdditiveAttention(video_dim, text_dim, hidden_dim)
        self.cross_gating = CrossGating(video_dim, text_dim)
        self.gru = nn.GRU(video_dim + text_dim, out_dim)

    @shape(video=('T', 'd_v'), text=('S', 'd_s'))
    def forward(self, video: Tensor, text: Tensor):
        """
        :return: shape=(T, out_dim)
        """
        T, d_v = video.shape
        S, d_s = text.shape
        h_s_bar = self.attention(video, text.unsqueeze(0).expand(T, S, d_s))
        h_v_tilde, h_s_tilde = self.cross_gating(video, h_s_bar)
        h_r = self.gru(torch.cat((h_v_tilde, h_s_tilde), dim=1).view(T, 1, d_v + d_s))[0].view(T, -1)
        return h_r


class CrossGating(nn.Module):
    def __init__(self, dim1: int, dim2: int):
        super().__init__()
        self.linear1 = nn.Linear(dim1, dim2)
        self.linear2 = nn.Linear(dim2, dim1)
        self.activation = nn.ReLU()

    @shape(x1=('N', 'dim1'), x2=('N', 'dim2'))
    def forward(self, x1: Tensor, x2: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :return:
            - shape=(N, dim1)
            - shape=(N, dim2)
        """
        x1 = self.activation(x1) * x2
        x2 = self.activation(x2) * x1
        return x1, x2


class SelfInteractor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attention = AdditiveAttention(dim, dim, dim)
        self.gru = nn.GRU(2 * dim, dim)

    @shape(X=('N', 'd'))
    def forward(self, X: Tensor) -> Tensor:
        """
        :return: shape=(N, d)
        """
        N, d = X.shape
        att = torch.stack([
            self.attention(X[i].view(1, -1), X[i:].view(1, N-i, d)).view(d) for i in range(N)
        ])
        gru_in = torch.cat((X, att), dim=1).unsqueeze(1)
        assert gru_in.shape == (N, 1, 2 * d)
        return self.gru(gru_in)[0].view(N, d)


class SegmentLocalizer(nn.Module):
    def __init__(self, text_dim: int, video_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.weight_pipeline = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
            nn.Flatten(),
            nn.Softmax()
        )
        self.cross_pipeline = nn.Sequential(
            nn.Linear(text_dim + video_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=False),
            nn.Flatten()
        )

    @shape(video=('T', 'd_v'), text=('N', 'd_s'))
    def forward(self, video: Tensor, text: Tensor):
        """
        :return: shape=(d,)
        """
        N, d = text.shape
        T = video.shape[0]
        weights = self.weight_pipeline(text).view(N, 1)
        h_o = torch.sum(text * weights, dim=0).expand(T, d)
        return self.cross_pipeline(torch.cat((video, h_o), dim=1))
