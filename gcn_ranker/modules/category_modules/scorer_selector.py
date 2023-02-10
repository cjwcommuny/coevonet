import math

import torch
from arraycontract import ndim
from torch import nn, Tensor
from torch.nn import init


class ScorerSelector(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, num_scorers: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.scorers = nn.Parameter(torch.empty(num_scorers, 1, embed_dim), requires_grad=True)
        self.embed_dim = embed_dim
        #
        self.reset_weights()

    def reset_weights(self):
        init.kaiming_uniform_(self.scorers, a=math.sqrt(5))

    @ndim(query=1, key=2)
    def forward(self, query: Tensor, key: Tensor):
        query = query.reshape(1, 1, -1)
        key = key.reshape(key.shape[0], 1, key.shape[1])
        scorer = self.attention(query, key, self.scorers)[0].reshape(-1)
        return scorer
