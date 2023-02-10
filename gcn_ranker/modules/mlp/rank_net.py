from typing import List

from torch import Tensor, nn
from torchtools.modules import MultiFullyConnectedLayer

class RankNet(nn.Module):
    def __init__(self, dims: List[int], dropout_rate: float):
        super().__init__()
        self.layers = nn.Sequential(
            MultiFullyConnectedLayer(dims, dropout_rate),
            nn.Linear(dims[-1], 1)
        )

    def forward(self, x: Tensor):
        """
        :return shape=(batch,)
        """
        return self.layers(x).squeeze(1)
