import os

import torch
from functionalstream import Stream
from torch import Tensor
from torch.utils.data import Dataset

from gcn_ranker.data.utils import str_to_tensor_1d
from gcn_ranker.data.youtube_highlight.features.utils import SAMPLING_STRATEGIES, load_feature, convert_labels
from gcn_ranker.data.youtube_highlight.youtube_highlight import YoutubeHighlight, load_match_labels, segment2str


class YoutubeHighlightSegmentFeatures(Dataset):
    def __init__(self, feature_folder_name: str, sample_strategy: str, *args, **kwargs):
        super().__init__()
        self.dataset = YoutubeHighlight(*args, **kwargs, loader=self.loader)
        self.feature_folder_name = feature_folder_name
        self.sample_strategy = SAMPLING_STRATEGIES[sample_strategy]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    @staticmethod
    def generate_segments_labels(video_dir: str):
        segments, labels = load_match_labels(os.path.join(video_dir, 'match_label.json'))
        segments = Stream(segments) \
            .map(lambda start, end: (int(start), int(end))) \
            .map(lambda segment: segment2str(segment)) \
            .to_list()
        return list(zip(segments, labels))

    def loader(self, video_dir: str):
        features_dir = os.path.join(video_dir, self.feature_folder_name)
        segments_labels = self.generate_segments_labels(video_dir)
        segments_labels = self.sample_strategy(segments_labels)
        features, labels = Stream(segments_labels)\
            .map(lambda segment, label: (os.path.join(features_dir, f'{segment}.txt'), label))\
            .filter(lambda feature_path, label: os.path.exists(feature_path))\
            .map(lambda feature_path, label: (load_feature(feature_path), label))\
            .unpack_tuples()
        features = torch.stack(features)
        labels = torch.tensor(labels)
        labels = convert_labels(labels)
        return features, labels

    @staticmethod
    def collate_fn(batches):
        assert len(batches) == 1
        return batches[0]

    @property
    def categories(self):
        return self.dataset.CATEGORIES