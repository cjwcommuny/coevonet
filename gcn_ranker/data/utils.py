import os
import random
import shutil
from typing import Optional, List, Generator

import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset, Subset
import numpy as np


def tensor_1d_to_str(arr: Tensor, ndigits: Optional[int]=None) -> str:
    process = lambda x: x if ndigits is None else round(x, ndigits)
    return ','.join([str(process(x.item())) for x in arr])

def str_to_tensor_1d(s: str) -> Tensor:
    lst = [float(x) for x in s.split(',')]
    return torch.tensor(lst)

def tensor_2d_to_str(arr: Tensor, ndigits: Optional[int]=None) -> str:
    return ';'.join([tensor_1d_to_str(arr_1d, ndigits) for arr_1d in arr])

def str_to_tensor_2d(s: str) -> Tensor:
    lst = [[float(x) for x in row.split(',')] for row in s.split(';')]
    return torch.tensor(lst)


def split_dataset(dataset: Dataset, num_replicas: int) -> Generator:
    indices = np.array_split(np.arange(len(dataset)), num_replicas)
    indices = [x.tolist() for x in indices]
    return (Subset(dataset, indice) for indice in indices)


def renew_dir(dir: str):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

class RatioSubset(Dataset):
    def __init__(self, dataset: Dataset, ratio: float):
        self.dataset = Subset(
            dataset,
            indices=random.sample(
                population=range(len(dataset)),
                k=int(ratio * len(dataset))
            )
        )
        self.ratio = ratio

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


def ratio_random_split(dataset: Dataset, first_ratio: float, seed: int):
    first_len = int(first_ratio*len(dataset))
    return torch.utils.data.random_split(
        dataset,
        lengths=[first_len, len(dataset) - first_len],
        generator=torch.Generator().manual_seed(seed)
    )
