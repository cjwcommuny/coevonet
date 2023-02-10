import os

import torch
from functionalstream import Stream
from torch import Tensor

from gcn_ranker.data.utils import str_to_tensor_1d
from gcn_ranker.data.youtube_highlight.youtube_highlight import segment2str, load_match_labels, YoutubeHighlight

SAMPLING_STRATEGIES = {
    'all': lambda x: x,
    'filter_normal': lambda lst:
    Stream(lst).filter(lambda segment, label: label != YoutubeHighlight.NORMAL).to_list()
}



def load_feature(path: str) -> Tensor:
    file = open(path, 'r')
    x = str_to_tensor_1d(file.read())
    file.close()
    return x


def convert_labels(labels: Tensor) -> Tensor:
    """
    -1 -> 0; 1 -> 1
    """
    return (labels * 0.5 + 0.5).to(dtype=torch.long)