import torch
from arraycontract import shape
from torch import Tensor, nn

class Video2Gif(nn.Module):
    """
    Gygli, M., Song, Y., & Cao, L. (2016). Video2gif: Automatic generation of animated gifs from video. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 1001-1009).
    """
    def __init__(self, in_features: int, *args, **kwargs):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    @shape(features=('N', 'd'))
    def forward(self, features: Tensor):
        return self.fc(features).flatten()

class HuberLoss(nn.Module):
    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta

    @staticmethod
    def l_p(p: float, positive: Tensor, negative: Tensor):
        return torch.max(torch.tensor(0.0).to(positive.device), 1 - positive + negative).pow(p)

    def forward(self, positive: Tensor, negative: Tensor):
        u = 1 - positive + negative
        if u <= self.delta:
            loss = 0.5 * self.l_p(2, positive, negative)
        else:
            loss = self.delta * self.l_p(1, positive, negative) - 0.5 * self.delta * self.delta
        return torch.mean(loss)
