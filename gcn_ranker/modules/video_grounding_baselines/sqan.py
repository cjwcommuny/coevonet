from collections import OrderedDict
from math import sqrt
from typing import Optional

import torch
from arraycontract import shape
from torch import Tensor, nn
from torchtools.tensors import has_nan, has_inf
from transformers import AutoModelWithLMHead
import torch.nn.functional as F

from gcn_ranker.metrics.smooth_l1 import smooth_l1_norm
from gcn_ranker.modules.common import View, NewAdditiveAttention

class Encoder(nn.Module):
    def __init__(
            self,
            vocab_size: Optional[int],
            word_embed_dim: Optional[int],
            pretrained_word_embedding: Optional[str],
            video_feature_dim: int,
            hidden_dim: int,
            dropout: float,
            max_len: int=200
    ):
        super().__init__()
        assert hidden_dim % 2 == 0
        self.segment_encoding = nn.Sequential(
            nn.Linear(video_feature_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True)
        )
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
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
            ('lstm', nn.LSTM(hidden_dim, hidden_dim // 2, bidirectional=True))
        ]))

    @shape(video=('T', 'd'), query=('S',))
    def forward(self, video: Tensor, query: Tensor):
        """
        :param video:
        :param query:
        :return:
            - F_v: shape=(T, d)
            - F_q: shape=(S, d)
            - q: shape=(d)
        """
        T, S = video.shape[0], query.shape[0]
        F_v = self.segment_encoding(video) + self.position_embedding(torch.arange(T, device=video.device))
        F_q, (q, _) = self.text_pipeline(query)
        F_q, q = F_q.view(S, -1), q.view(-1)
        return F_v, F_q, q


class SQAN(nn.Module):
    def __init__(self, dim: int, W_g: nn.Module, additive_attention: nn.Module):
        super().__init__()
        self.W_q = nn.Linear(dim, dim)
        self.W_g = W_g # shared
        self.additive_attention = additive_attention # shared
        #
        self.dim = dim

    @shape(e=('d',), q=('d',), E=('N', 'd'))
    def forward(self, e: Tensor, q: Tensor, E: Tensor):
        """
        :param e:
        :param q: sentence feature
        :param E: words features
        :return:
        """
        e, q, E = e.unsqueeze(0), q.unsqueeze(0), E.unsqueeze(0)
        g = torch.relu(self.W_g(torch.cat([self.W_q(q), e], dim=1)))
        assert g.shape == (1, self.dim)
        new_e, a = self.additive_attention(g, E)
        new_e, a = new_e.squeeze(0), a.squeeze(0)
        return new_e, a

class SQANs(nn.Module):
    def __init__(self, dim: int, num_SQAN: int):
        super().__init__()
        self.W_g = nn.Linear(2 * dim, dim)
        self.additive_attention = NewAdditiveAttention(dim, dim, dim // 2)
        self.sqans = nn.ModuleList(
            [SQAN(dim, self.W_g, self.additive_attention)] * num_SQAN
        )
        #
        self.dim = dim

    @shape(q=('dim',), E=('N', 'd'))
    def forward(self, q: Tensor, E: Tensor):
        """
        :param q:
        :param E:
        :return:
            - shape=(num_SQAN, d)

        """
        es = [torch.zeros(self.dim, device=q.device)]
        A = []
        for sqan in self.sqans:
            e, a = sqan(es[-1], q, E)
            es.append(e)
            A.append(a)
        es = es[1:]
        return torch.stack(es), torch.stack(A)


class ISPU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.W_s = nn.Linear(dim, dim)
        self.W_e = nn.Linear(dim, dim)
        self.W_m = nn.Linear(dim, dim)
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=15, padding=7),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, kernel_size=15, padding=7)
        )


    @shape(S=('T', 'd'), e=('d',))
    def forward(self, S: Tensor, e: Tensor):
        T, d = S.shape
        m_tilde = self.W_m(self.W_s(S) * self.W_e(e.unsqueeze(0))).t().unsqueeze(0)
        assert m_tilde.shape == (1, d, T)
        M = self.conv1d(m_tilde) + m_tilde
        M = M.squeeze(0).t()
        assert M.shape == (T, d)
        return M

class LGVTI(nn.Module):
    def __init__(self, num_block: int, dim: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ISPU(dim)] * num_block
        )
        self.c_satt = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 1),
            nn.Softmax(dim=0)
        )
        self.none_local_block = NLBlock(dim)
        #
        self.num_block = num_block

    @shape(S=('T', 'd'), es=('num_block', 'd'))
    def forward(self, S: Tensor, es: Tensor):
        T, d = S.shape
        M = torch.stack([block(S, e) for e, block in zip(es, self.blocks)])
        assert M.shape == (self.num_block, T, d)
        #
        c = self.c_satt(es).view(self.num_block, 1, 1)
        R_tilde = torch.sum(c * M, dim=0)
        assert R_tilde.shape == (T, d)
        R = self.none_local_block(R_tilde)
        return R


class NLBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.W_rq = nn.Linear(dim, dim)
        self.W_rk = nn.Linear(dim, dim)
        self.W_rv = nn.Linear(dim, dim)
        #
        self.dim = dim
        self.dim_sqrt = sqrt(dim)

    @shape(R=('T', 'd'))
    def forward(self, R: Tensor):
        """
        :param R:
        :return: shape=(T, d)
        """
        T, d = R.shape
        TT = torch.mm(self.W_rk(R), self.W_rq(R).t()) / self.dim_sqrt
        assert TT.shape == (T, T)
        return R + torch.mm(torch.softmax(TT, dim=1), self.W_rv(R))


class TABR(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp_tatt = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 1)
        )
        self.mlp_reg = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 2)
        )

    @shape(R=('T', 'd'))
    def forward(self, R: Tensor):
        r = self.mlp_tatt(R)
        o = torch.softmax(r, dim=0)
        v = torch.sum(o * R, dim=0, keepdim=True)
        t = self.mlp_reg(v).squeeze(0)
        return t, r


def tag_loss(r: Tensor, segment: Tensor):
    """
    :param r: shape=(N,)
    :param segment: shape=(2,)
    :return:
    """
    indices = torch.arange(r.shape[0], device=r.device)
    mask = (indices >= segment[0]) * (indices < segment[1])
    r_inside_segment = r.masked_select(mask)
    loss = - torch.mean(F.log_softmax(r_inside_segment, dim=0))
    return loss


def dqa_loss(A: Tensor, Lambda: float) -> Tensor:
    I = torch.eye(A.shape[0], device=A.device)
    loss = torch.norm(torch.mm(A, A.t()) - Lambda * I).square()
    return loss

class LGVTITG(nn.Module):
    """
    J. Mun, M. Cho, and B. Han, “Local-Global Video-Text Interactions for Temporal Grounding,” pp. 10810–10819, 2020.
    """
    def __init__(
            self,
            window_size: int,
            vocab_size: Optional[int],
            word_embed_dim: Optional[int],
            pretrained_word_embedding: Optional[str],
            video_feature_dim: int,
            hidden_dim: int,
            dropout: float,
            num_SQAN: int,
            dqa_lambda: float,
            tag_loss_ratio: float,
            dqa_loss_ratio: float
    ):
        super().__init__()
        self.window_size = window_size
        self.encoder = Encoder(
            vocab_size, word_embed_dim, pretrained_word_embedding, video_feature_dim, hidden_dim, dropout)
        self.sqans = SQANs(hidden_dim, num_SQAN)
        self.lgvti = LGVTI(num_SQAN, hidden_dim)
        self.tabr = TABR(hidden_dim)
        #
        self.dqa_lambda = dqa_lambda
        self.tag_loss_ratio = tag_loss_ratio
        self.dqa_loss_ratio = dqa_loss_ratio

    @shape(video=('T', 'd'), query=('S',), ground_truth=())
    def forward(self, video: Tensor, query: Tensor, ground_truth: Optional[Tensor]):
        S, E, q = self.encoder(video, query)
        es, A = self.sqans(q, E)
        R = self.lgvti(S, es)
        position, r = self.tabr(R)
        #
        if ground_truth is None:
            return position
        ground_truth = ground_truth.item()
        ground_truth = torch.tensor([ground_truth, ground_truth + self.window_size], device=video.device)
        location_regression_loss = smooth_l1_norm(position[0] - ground_truth[0]) # only regress start
        temporal_attention_guidance_loss = tag_loss(r, ground_truth)
        distinct_query_attention_loss = dqa_loss(A, self.dqa_lambda)
        total_loss = location_regression_loss + self.tag_loss_ratio * temporal_attention_guidance_loss + self.dqa_loss_ratio * distinct_query_attention_loss
        return position, total_loss, location_regression_loss, temporal_attention_guidance_loss, distinct_query_attention_loss


