from typing import Tuple

from torch import Tensor
import torch
from torchtools.tensors import mask_to_index_1d


def get_contrastive_pairs(scores: Tensor, labels: Tensor) -> Tensor:
    positive_scores = scores[mask_to_index_1d(labels == 1)].view(-1)
    negative_scores = scores[mask_to_index_1d(labels == 0)].view(-1)
    score_pairs = torch.cartesian_prod(positive_scores, negative_scores)
    return score_pairs

def pairwise_acc(pairs: Tensor) -> Tuple[int, int]:
    return torch.sum(pairs[:,0] > pairs[:,1]).item(), pairs.shape[0]
