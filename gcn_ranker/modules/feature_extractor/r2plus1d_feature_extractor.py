from typing import Optional, List

import torch
from arraycontract import shape
from torch import Tensor, nn
from torchvision.models.video.resnet import r2plus1d_18

# from gcn_ranker.modules.common import normalize_batched_video
from gcn_ranker.modules.common import normalize_batched_video

_model_map = {
    'r2plus1d_18': r2plus1d_18 # out_dim=512
}

class VideoResnetFeatureExtractor(nn.Module):
    MEAN = [0.43216, 0.394666, 0.37645]
    STD = [0.22803, 0.22145, 0.216989]
    IMAGE_SIZE = (112, 112)

    def __init__(self, arch: str, model_path: Optional[str]=None, need_normalize: bool=True):
        super().__init__()
        should_download = model_path is None
        extractor = _model_map[arch](pretrained=should_download)
        if not should_download:
            extractor.load_state_dict(torch.load(model_path))
        self.extractor = nn.Sequential(
            extractor.stem,
            extractor.layer1,
            extractor.layer2,
            extractor.layer3,
            extractor.layer4,
            extractor.avgpool
        ) # output.shape=(batch_size, 512, 1, 1, 1)
        #
        self.need_normalize = need_normalize


    @shape(frames=('batch_size', 'T', 3, *IMAGE_SIZE))
    def forward(self, frames: Tensor) -> Tensor:
        """
        :return feature.shape=(batch_size, dim)
        """
        batch_size = frames.shape[0]
        if self.need_normalize:
            frames = normalize_batched_video(frames, self.MEAN, self.STD)
        frames = frames.transpose(1, 2)
        assert frames.shape[1] == 3 and frames.shape[3:] == (112, 112)
        feature = self.extractor(frames).view(batch_size, -1)
        return feature
