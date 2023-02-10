import json
import os

from typing import Callable, List, Optional, Union, Any, Tuple

from torch.utils.data.dataset import Dataset

from functionalstream import Stream
from pybetter.os import list_folder

from cached_property import cached_property

def load_match_labels(path: str) -> Tuple[List[List[int]], List[int]]:
    segments, labels = json.load(open(path, 'r'))
    assert len(segments) == len(labels)
    return segments, labels

def segment2str(segment: Tuple[int, int]) -> str:
    return f'{segment[0]}-{segment[1]}'

class YoutubeHighlight(Dataset):
    CATEGORIES = ('dog', 'gymnastics', 'parkour', 'skating', 'skiing', 'surfing')
    VIDEO_EXTENSIONS = ('.mkv', '.mp4')
    SPLIT2IDX = {
        'train': (0, 1),
        'val': (2, 3),
        'all': (0, 1, 2, 3)
    }
    HIGHLIGHT, NORMAL, NON_HIGHLIGHT = 1, 0, -1

    def __init__(
            self,
            root: str,
            split: str,
            loader: Callable[[str], Union[tuple, Any]],
            is_valid: Optional[Callable[[str], bool]]=None
    ):
        """
        :param root:
        :param split: in {'train', 'val', 'all'}
        :param loader:
        :param is_valid:
        """
        super().__init__()
        self.root = root
        self.split = split
        assert self.split in self.SPLIT2IDX
        self.loader = loader
        self.is_valid = is_valid if is_valid is not None \
            else self.video_valid_checker

    def __len__(self):
        return len(self.indexes)

    @staticmethod
    def get_video_names(file_name: str, categories_dir: Stream, split_idxes: Tuple[int, ...]) -> Stream:
        return categories_dir \
            .map(lambda _, category_dir: os.path.join(category_dir, file_name)) \
            .map(lambda vlist_path: json.load(open(vlist_path, 'r'))) \
            .map(lambda vlist: Stream(split_idxes).map(lambda idx: vlist[idx][0]).flatten().to_list()) \
            .flatten()

    @staticmethod
    def video_valid_checker(folder_path: str) -> bool:
        video_id = os.path.basename(folder_path)
        return any(
            os.path.exists(
                os.path.join(folder_path, f'{video_id}{extension}')
            )
            for extension in YoutubeHighlight.VIDEO_EXTENSIONS
        )


    @cached_property
    def indexes(self) -> List[Tuple[str, str, str]]:
        categories_dir = Stream(self.CATEGORIES)\
            .map(lambda category: (category, os.path.join(self.root, category)))
        vlist_video_names = self.get_video_names('vlist.json', categories_dir, self.SPLIT2IDX[self.split])
        vlist_sel_video_names = self.get_video_names('vlist_sel.json', categories_dir, self.SPLIT2IDX[self.split])
        video_names = set().union(vlist_video_names, vlist_sel_video_names)
        #
        indexes = categories_dir \
            .map(lambda category, category_dir: (category, category_dir, list_folder(category_dir))) \
            .map(
            lambda category, category_dir, video_ids:
            [(category, id, os.path.join(category_dir, id)) for id in video_ids]
        ) \
            .flatten() \
            .filter(lambda category, video_id, video_dir: video_id in video_names) \
            .filter(lambda category, video_id, video_dir: self.is_valid(video_dir)) \
            .to_list()
        return indexes

    @cached_property
    def video_id2index(self):
        return {video_id: idx for idx, (category, video_id, video_dir) in enumerate(self.indexes)}

    def __getitem__(self, item):
        idx = self.video_id2index[item] if type(item) == str else item
        category, video_id, video_dir = self.indexes[idx]
        data = self.loader(video_dir)
        if isinstance(data, tuple):
            return (video_id, category, *data)
        else:
            return video_id, category, data


