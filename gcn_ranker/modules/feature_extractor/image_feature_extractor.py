from torch import nn
from torchtools.modules import ResNetFeatureExtractor

_model_map = {
    'resnet50': ResNetFeatureExtractor
}

class ImageFeatureExtractor(nn.Module):
    IMAGE_SIZE = 224

    def __init__(self, arch: str, *args, **kwargs):
        super().__init__()
        self.extractor = _model_map[arch](*args, **kwargs)

    def forward(self, x):
        return self.extractor(x)
