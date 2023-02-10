import os

import cv2
from cached_property import cached_property
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Any

from PIL import Image
from functionalstream import Stream
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from gcn_ranker.data.youtube_highlight.youtube_highlight import YoutubeHighlight


class YoutubeHighlightFrames(Dataset):
    def __init__(
            self,
            in_folder_name: str,
            image_opener: str,
            image_transform: Optional[Callable]=None,
            *args,
            **kwargs
    ):
        super().__init__()
        self.dataset = YoutubeHighlight(*args, **kwargs, loader=None)
        self.in_folder_name = in_folder_name
        self.image_transform = image_transform
        self.read_image = {'pil': Image.open, 'opencv': cv2.imread}[image_opener]

    @cached_property
    def indexes(self) -> List[Tuple[str, str, int, str]]:
        """
        :return [(category, video_id, frame_idx, path)]
        """
        return Stream(self.dataset.indexes)\
            .map(lambda category, video_id, video_dir: (video_dir, os.path.join(video_dir, 'frames')))\
            .map(lambda video_dir, frames_dir: [
                (video_dir, int(Path(frame_file_name).stem), os.path.join(frames_dir, frame_file_name))
                for frame_file_name in os.listdir(frames_dir)
            ])\
            .flatten()\
            .to_list()

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx) -> Tuple[str, int, Any]:
        video_dir, frame_idx, frame_path = self.indexes[idx]
        try:
            frame = self.read_image(frame_path)
            if self.image_transform is not None:
                frame = self.image_transform(frame)
            return video_dir, frame_idx, frame
        except Exception as e:
            print(f'ERROR: {video_dir}, {frame_path}')
            raise e
