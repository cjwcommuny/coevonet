import copy
from collections import Callable
from typing import List

import torch
from torch import nn, Tensor

class NodeAttention(nn.Module):
    def __init__(self, h_dim: int, V_dim: int, hidden_dim: int, activation: Callable):
        super().__init__()
        self.h_project = nn.Linear(h_dim, hidden_dim)
        self.V_project = nn.Linear(V_dim, hidden_dim, bias=False)
        self.activation = activation

    def forward(self, h: Tensor, V: Tensor):
        """
        :param h: shape=(h_dim,)
        :param V: shape=(V_len, V_dim)
        :return: shape=(V_len, hidden_dim)
        """
        h = self.h_project(h.view(1, -1))
        V = self.V_project(V)
        return self.activation(h + V)


class GeneralSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, *args):
        for layer in self.layers:
            args = layer(*args)
        return args


class Loop(nn.Module):
    def __init__(self, module: nn.Module, n_times: int, share: bool):
        super().__init__()
        self.module = module
        self.layers = GeneralSequential(*[
            module if share else copy.deepcopy(module) for _ in range(n_times)
        ])

    def forward(self, *args):
        return self.layers(*args)


def normalize_batched_video(tensor: Tensor, mean: List[float], std: List[float], inplace: bool=False):
    """
    :param tensor: shape=(batch_size, T, C, H, W)
    :param mean:
    :param std:
    :param inplace:
    """
    assert tensor.ndim == 5
    if not inplace:
        tensor = tensor.clone()
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(
            'std evaluated to zero after conversion to {}, leading to division by zero.'.format(tensor.dtype)
        )
    if mean.ndim == 1:
        mean = mean[None, None, :, None, None]
    if std.ndim == 1:
        std = std[None, None, :, None, None]
    tensor.sub_(mean).div_(std)
    return tensor


def index2mask_1d(index: Tensor, N: int):
    assert index.ndim == 1
    mask = torch.zeros(N).type_as(index)
    mask.scatter_(dim=0, index=index, value=1)
    return mask


class View(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return x.view(*self.args, **self.kwargs)


class Expand(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return x.expand(*self.args, **self.kwargs)


class NewAdditiveAttention(nn.Module):
    def __init__(
            self,
            h_len,
            v_len,
            hidden_dim
    ):
        super().__init__()
        self.W = nn.Linear(h_len, hidden_dim)
        self.U = nn.Linear(v_len, hidden_dim, bias=False)
        self.tanh = nn.Tanh()
        self.w = nn.Linear(hidden_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        #
        self.out_feature_dim = v_len


    def forward(self, h, V):
        """
        apply attention from h to V

        e = w^T * tanh(W * h + U * V^T + b)
        beta = softmax(e)
        result = weight_sum(weight=beta, V)

        :param h: shape=(batch_size, h_len)
        :param V: shape=(batch_size, seq_len, v_len)
        :return:
            - result: shape=(batch_size, v_len)
            - beta: shape=(batch_size, seq_len)
        """
        batch_size, h_len = h.shape
        seq_len, v_len = V.shape[1:]
        assert batch_size == V.shape[0]
        assert h_len == self.W.in_features, f"{h.shape}, {self.W.in_features}"
        assert v_len == self.U.in_features, f"{V.shape}, {self.U.in_features}"

        h_after_projection = self.W(h).unsqueeze(dim=1) # shape=(batch_size, 1, hidden_dim)
        V_after_projection = self.U(V) # shape=(batch_size, seq_len, hidden_dim)
        e = self.w(self.tanh(h_after_projection + V_after_projection)).view(batch_size, seq_len)
        beta = self.softmax(e) # shape=(batch_size, seq_len)
        result = torch.einsum("bs,bsv->bv", beta, V)
        return result, beta

class Rescale(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return self.scale * x

class Empty(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args):
        return args if len(args) > 1 else args[0]
