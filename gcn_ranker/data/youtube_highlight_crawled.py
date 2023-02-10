import os
from pathlib import Path
from typing import Tuple, List, Dict, Callable

from functionalstream import Stream

from torch.utils.data.dataset import Dataset
from torchvision.datasets import DatasetFolder


def parse_duration_statistics_file(path: str) -> List[Tuple[str, int]]:
    return Stream(open(path, 'r').readlines())\
        .map(lambda line: tuple(line.split()))\
        .map(lambda video_id, duration: (video_id, int(duration)))\
        .to_list()


def parse_duration_statistics_files(dir: str) -> Dict[str, Tuple[str, int]]:
    """
    :return {video_id: (category, duration)}
    """
    return Stream(os.listdir(dir))\
        .map(lambda file_name: (Path(file_name).stem, os.path.join(dir, file_name)))\
        .map(lambda category, path: (category, parse_duration_statistics_file(path)))\
        .map(lambda category, durations: [(video_id, category, duration) for video_id, duration in durations])\
        .flatten()\
        .collect(lambda stream: {video_id: (category, duration) for video_id, category, duration in stream})






class YoutubeCrawled(Dataset):
    def __init__(self, root: str, durations_dir: str, loader: Callable):
        super().__init__()
        self.dataset = DatasetFolder(root=root, loader=loader)
        self.id2category_duration = parse_duration_statistics_files(durations_dir)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        video_id, data = self.dataset[idx]
        category, duration = self.id2category_duration[video_id]
        return video_id, category, data
