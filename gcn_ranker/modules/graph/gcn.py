import math
from typing import List

import torch
from arraycontract import shape
from torch import nn as nn, Tensor
from torch.nn import Parameter


class GraphConvolutionLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, add_identity: bool=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.add_identity = add_identity
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: Tensor, adj: Tensor):
        """
        :param input: shape=(node_num, feature_dim)
        :param adj: shape=(node_num, node_num)
        :return:
        """
        node_num = input.shape[0]
        if self.add_identity:
            adj = adj + torch.eye(node_num, dtype=adj.dtype, device=adj.device)
        degrees_sqrt_reciprocal = torch.diag(torch.reciprocal(torch.sqrt(torch.sum(adj, dim=1))))
        support = torch.mm(input, self.weight)
        adj_normalized = torch.mm(torch.mm(degrees_sqrt_reciprocal, adj), degrees_sqrt_reciprocal)
        output = torch.mm(adj_normalized, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MultiLayerGcn(nn.Module):
    def __init__(self, dims: List[int], dropout: float, bias: bool=True, add_identity: bool=True):
        super().__init__()
        self.gcns = nn.ModuleList(
            [GraphConvolutionLayer(in_dim, out_dim, bias, add_identity)
             for in_dim, out_dim in zip(dims[:-1], dims[1:])]
        )
        self.dropout = nn.Dropout(dropout)

    @shape(input=('N','d'), adj=('N', 'N'))
    def forward(self, input: Tensor, adj: Tensor):
        x = input
        for layer in self.gcns:
            x = torch.relu(self.dropout(layer(x, adj)))
        return x
