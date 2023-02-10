import torch
from torch import Tensor

def smooth_l1_norm(x: Tensor) -> Tensor:
    x_abs = torch.abs(x)
    return 0.5 * torch.square(x) if x_abs < 1 else x_abs - 0.5
