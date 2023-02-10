from typing import Tuple

import torch
from arraycontract import shape
from torch import nn, Tensor
import torch.nn.functional as F

class WeightSelector(nn.Module):
    def __init__(self, num_keys: int, out_shape: Tuple[int,...]):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(num_keys, *out_shape), requires_grad=True)

    @shape(query=('d',), key=('num_keys', 'd'))
    def forward(self, query: Tensor, key: Tensor):
        query = query.unsqueeze(dim=0) # shape=(1, d)
        attention = F.softmax(F.cosine_similarity(query, key), dim=0) # shape=(num_keys,)
        attention = attention.expand(self.weights.shape)
        result = torch.sum(attention * self.weights, dim=0)
        return result # shape=out_shape


