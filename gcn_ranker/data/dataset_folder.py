import json
import os
from typing import Callable, Optional, Tuple, List, Dict, Collection, Set

from torch.utils.data import Dataset


class DatasetFolderOfFolder(Dataset):
    """
    root/class_x/id1/
    root/class_x/id2/
    ...
    root/class_y/id10/
    ...
    """
    def __init__(
            self,
            root: str,
            loader: Callable,
            transform: Optional[Callable]=None,
            target_transform: Optional[Callable]=None,
            is_valid_folder: Optional[Callable[[str], bool]]=None,
            filter_classes: Optional[Collection[str]]=None,
            filter_folders: Optional[Collection[str]]=None,
            return_index: bool=False,
            indexes: Optional[List[Tuple[int, str]]]=None
    ):
        super().__init__()
        self.classes, self.class_to_idx = self._find_classes(root)
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.return_index = return_index
        #
        if indexes is None:
            samples = self.make_dataset(
                root,
                self.classes,
                is_valid_folder if is_valid_folder is not None else lambda x: True,
                self.class_to_idx,
                (lambda x: x in set(filter_folders)) if filter_folders is not None else (lambda x: True)
            ) # [(folder_path, cls_idx)]
            filter_classes = set(self.classes) if filter_classes is None else set(filter_classes)
            self.samples = [
                (folder_path, cls_idx)
                for folder_path, cls_idx in samples
                if self.classes[cls_idx] in filter_classes
            ]
        else:
            self.samples = [
                (os.path.join(root, cls, str(id)), self.class_to_idx[cls])
                for id, cls in indexes
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_index:
            return index, sample, target
        else:
            return sample, target

    @staticmethod
    def _find_classes(dir: str) -> Tuple[List[str], Dict[str, int]]:
        classes = [cls for cls in os.listdir(dir) if os.path.isdir(os.path.join(dir, cls))]
        classes.sort()
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        return classes, class_to_idx

    @staticmethod
    def make_dataset(
            root: str,
            classes: List[str],
            is_valid_folder: Callable[[str], bool],
            class_to_idx: Dict[str, int],
            filter_folder: Callable[[str], bool]
    ) -> List[Tuple[str, int]]:
        return [
            (os.path.join(root, cls, folder_name), class_to_idx[cls])
            for cls in classes
            for folder_name in os.listdir(os.path.join(root, cls))
            if is_valid_folder(os.path.join(root, cls, folder_name)) and filter_folder(folder_name)
        ]


class PredefinedDatasetFolder(Dataset):
    def __init__(self, root: str, index: str, loader: Callable):
        """
        :param root:
        :param index: json file format like [(folder_name: str, class: str)]
        :param loader:
        """
        super().__init__()
        self.loader = loader
        index = json.load(open(index, 'r'))
        self.classes = sorted(list(set(cls for folder_name, cls in index)))
        self.class2index = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = [
            (os.path.join(root, cls, folder_name), self.class2index[cls])
            for folder_name, cls in index
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        return sample, target
