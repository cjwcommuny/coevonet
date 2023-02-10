import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import os

import torch
from functionalstream.stream import Stream
from torch.utils.data.dataset import Dataset
from torchtools.tensors import mask_to_index_1d

from gcn_ranker.data.youtube_highlight.features.utils import SAMPLING_STRATEGIES, convert_labels
from gcn_ranker.data.youtube_highlight.youtube_highlight import YoutubeHighlight, load_match_labels


class YoutubeHighlightFrameFeatures(Dataset):
    def __init__(self, feature_folder_name: str, sample_strategy: str, sample_step: int=1, *args, **kwargs):
        super().__init__()
        self.feature_folder_name = feature_folder_name
        self.dataset = YoutubeHighlight(*args, **kwargs, loader=self.loader)
        self.sample_strategy = SAMPLING_STRATEGIES[sample_strategy]
        self.sample_step = sample_step

    def __len__(self):
        return len(self.dataset)

    def loader(self, video_dir):
        feature_dir = os.path.join(video_dir, self.feature_folder_name)
        features = torch.load(os.path.join(feature_dir, 'frame_features.pkl'))
        frame_idxes = torch.load(os.path.join(feature_dir, 'frame_idxes.pkl'))
        sample_idxes = torch.arange(start=0, end=frame_idxes.shape[0], step=self.sample_step)
        features, frame_idxes = features[sample_idxes], frame_idxes[sample_idxes]
        #
        segment_labels = list(zip(*load_match_labels(os.path.join(video_dir, 'match_label.json'))))
        segment_labels = self.sample_strategy(segment_labels)
        segments, labels = Stream(segment_labels)\
            .map(lambda segment, label: (
                    mask_to_index_1d((segment[0] <= frame_idxes) * (frame_idxes <= segment[1])).view(-1),
                    label
                )
            )\
            .filter(lambda idxes, label: idxes.numel() != 0)\
            .unpack_tuples()
        labels = torch.tensor(labels, dtype=torch.long)
        labels = convert_labels(labels)
        return features, segments, labels


    def __getitem__(self, item):
        return self.dataset[item]

    @staticmethod
    def collate_fn(batches):
        assert len(batches) == 1
        return batches[0]
