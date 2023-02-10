from torch import Tensor
import torch

def self_loop(n: int) -> Tensor:
    return torch.diag(torch.ones(n))

def forward_graph(n: int) -> Tensor:
    return torch.diag(torch.ones(n - 1), 1)

def backward_graph(n: int) -> Tensor:
    return torch.diag(torch.ones(n - 1), -1)

def bidirectional_graph(n: int) -> Tensor:
    return forward_graph(n) + backward_graph(n)

def fully_connected_graph(n: int) -> Tensor:
    return torch.ones(n, n) - torch.diag(torch.ones(n))
