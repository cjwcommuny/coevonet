import torch
from arraycontract import shape
from torch import nn, Tensor
import torch.nn.functional as F

from gcn_ranker.modules.common import Expand

class AttFMWrapper(nn.Module):
    def __init__(self, window_size: int):
        super().__init__()
        self.model = AttFM()
        self.window_size = window_size

    @shape(video=('T', 'C', 'H', 'W'))
    def forward(self, video: Tensor, text: Tensor):
        T = video.shape[0]
        window_starts = torch.arange(start=0, end=T - self.window_size + 1, step=1).to(video.device)
        scores = torch.cat([self.model(video[start:start+self.window_size]) for start in window_starts])
        return scores, window_starts


class AttFM(nn.Module):
    """
    Jiao, Yifan, et al. “Three-dimensional attention-based deep ranking model for video highlight detection.” IEEE Transactions on Multimedia 20.10 (2018): 2693-2705.
    """
    IMAGE_SIZE = 224

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            FeatureModule(),
            AttentionModule(),
            RankingModule()
        )

    @shape(frames=('T', 'C', 'H', 'W'))
    def forward(self, frames: Tensor):
        frames = F.interpolate(frames, size=(self.IMAGE_SIZE, self.IMAGE_SIZE))
        return self.layers(frames)


class FeatureModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, frames: Tensor):
        return self.layers(frames)


class RankingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    @shape(features=(256,))
    def forward(self, features: Tensor):
        """
        :return: shape=(1,)
        """
        return self.layers(features.view(1, 256)).view(-1)


class AttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
            Expand(-1, 256, 6, 6)
        )

    @shape(features=('T', 256, 6, 6))
    def forward(self, features: Tensor):
        weights = self.layers(features)
        features = weights * features
        features = features\
            .sum(dim=0, keepdim=True)\
            .sum(dim=2, keepdim=True)\
            .sum(dim=3, keepdim=True)\
            .view(-1)
        return features
