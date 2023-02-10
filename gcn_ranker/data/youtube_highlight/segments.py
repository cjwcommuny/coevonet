import os
from cached_property import cached_property
from pathlib import Path
from typing import Callable, List, Tuple, Optional

from PIL import Image
from functionalstream import Stream
from functionalstream.stream import OneOffStream
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from gcn_ranker.data.youtube_highlight.youtube_highlight import YoutubeHighlight, load_match_labels


class YoutubeHighlightSegments(Dataset):
    def __init__(self, in_folder_name: str, image_transform: Callable, *args, **kwargs):
        super().__init__()
        self.dataset = YoutubeHighlight(*args, **kwargs, loader=None)
        self.in_folder_name = in_folder_name
        self.image_transform = image_transform

    @cached_property
    def indexes(self) -> List[Tuple[str, List[str], Tuple[float, float]]]:
        return Stream(self.dataset.indexes)\
            .map(lambda category, video_id, video_dir: (video_dir, self.load_segments(video_dir)))\
            .map(lambda video_dir, segments: [(video_dir, s) for s in segments])\
            .flatten()\
            .map(lambda video_dir, segment: (
                video_dir, os.path.join(video_dir, self.in_folder_name), segment)
            )\
            .map(lambda video_dir, frames_dir, segment:
                         (video_dir, [os.path.join(frames_dir, frame) for frame in os.listdir(frames_dir)], segment)
            )\
            .map(lambda video_dir, frame_paths, segment:
                         (
                             video_dir,
                             [path for path in frame_paths if segment[0] <= int(Path(path).stem) <= segment[1]],
                             segment
                         )
            )\
            .filter(lambda video_dir, paths, segment: len(paths) >= 1)\
            .to_list()

    @staticmethod
    def load_segments(video_dir: str) -> List[List[int]]:
        return load_match_labels(os.path.join(video_dir, 'match_label.json'))[0]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx) -> Optional[Tuple[str, Tuple[int, int], Tensor]]:
        video_dir, paths, (start, end) = self.indexes[idx]
        try:
            frames = OneOffStream(paths)\
                .map(lambda path: Image.open(path))\
                .map(lambda image: self.image_transform(image))\
                .map(lambda image: to_tensor(image))\
                .stack_to_tensor()
            return video_dir, (int(start), int(end)), frames
        except Exception as e:
            print(f'{video_dir}, {paths}, {(start, end)}')
            raise e