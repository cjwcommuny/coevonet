from typing import Callable, List, Tuple, Optional

from PIL import Image
from datascience.opencv import VideoCaptureIter, get_fps
from functionalstream.stream import OneOffStream

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchtools.images import opencv_image_to_torch_tensor
from torchvision import transforms
from pathlib import Path
import os
import cv2
import numpy as np
import pandas as pd
from torchvision.transforms.functional import to_tensor, resize, center_crop, normalize

from gcn_ranker.data.dataset_folder import DatasetFolderOfFolder
import torchtools

def is_valid_folder(path: str) -> bool:
    video_id: str = os.path.basename(os.path.normpath(path))
    return os.path.exists(os.path.join(path, f'{video_id}.mp4'))

def need_extract_features(feature_save_name: str):
    def func(path: str) -> bool:
        return is_valid_folder(path) and not os.path.isfile(os.path.join(path, feature_save_name))
    return func

def video_loader(path: str) -> cv2.VideoCapture:
    video_id: str = os.path.basename(os.path.normpath(path))
    return cv2.VideoCapture(os.path.join(path, f'{video_id}.mp4'))

def video_frames_loader(
        second_interval: Optional[float],
        return_fps: bool=False,
        resize_shape: int=256,
        crop_shape: int=224
):
    def loader(path: str):
        video = video_loader(path)
        fps = get_fps(video)
        frame_interval = 1 if second_interval is None else int(fps * second_interval)
        frames = OneOffStream(VideoCaptureIter(video))\
            .enumerate()\
            .filter(lambda idx, frame: idx % frame_interval == 0)\
            .map(lambda idx, frame: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))\
            .map(lambda frame: Image.fromarray(frame))\
            .map(lambda frame: resize(frame, resize_shape))\
            .map(lambda frame: center_crop(frame, crop_shape))\
            .map(lambda frame: to_tensor(frame))\
            .stack_to_tensor()
        return (frames, fps) if return_fps else frames
    return loader

def FramesLoader(frames_folder: str) -> Callable:
    def loader(path: str):
        frames_folder_path = os.path.join(path, frames_folder)
        return OneOffStream(os.listdir(frames_folder_path))\
            .map(lambda file_name:
                         (int(Path(file_name).stem), os.path.join(frames_folder_path, file_name))
            )\
            .sorted(key=lambda x: x[0])\
            .map(lambda frame_idx, frame_path: Image.open(frame_path))\
            .to_list()
    return loader


class FramesTransform:
    def __init__(self, size: Tuple[int, int]):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])

    def __call__(self, frames: List[Image.Image]) -> Tensor:
        """
        :param frames:
        :return: shape=(T,C,H,W)
        """
        return torch.stack([self.transform(frame) for frame in frames])


class TaobaoFeatureLoader:
    def __init__(self, features_file_name: str, meta_file_name: str, tokenizer: Optional[Callable]):
        super().__init__()
        self.features_file_name = features_file_name
        self.meta_info_loader = MetaInfoLoader(meta_file_name, tokenizer)

    def __call__(self, folder_path: str) -> Tuple[int, str, Tensor, Tensor, Tensor]:
        features: Tensor = torch.from_numpy(np.load(os.path.join(folder_path, self.features_file_name)))
        video_id, category, text, highlight_start = self.meta_info_loader(folder_path)
        return video_id, category, features, text, highlight_start

    @staticmethod
    def collate_fn(batches: List[tuple]):
        assert len(batches) == 1
        return batches[0]


class FramesPackedLoader:
    def __init__(self, file_name: str, meta_file_name: str, tokenizer: Optional[Callable]):
        super().__init__()
        self.file_name = file_name
        self.meta_info_loader = MetaInfoLoader(meta_file_name, tokenizer)

    def __call__(self, folder_path: str):
        frames = torch.from_numpy(np.load(os.path.join(folder_path, self.file_name)))
        video_id, category, text, highlight_start = self.meta_info_loader(folder_path)
        return video_id, category, frames, text, highlight_start
        

class MetaInfoLoader:
    def __init__(self, meta_file_name: str, tokenizer: Optional[Callable]):
        super().__init__()
        self.meta_file_name = meta_file_name
        self.tokenizer = tokenizer

    def __call__(self, folder_path: str):
        meta_info: pd.Series = pd.read_csv(os.path.join(folder_path, self.meta_file_name), sep='\t').iloc[0]
        (video_id, text, highlight_start, category) \
            = meta_info['video_id'], meta_info['title'], meta_info['start'], meta_info['super_cate']
        if self.tokenizer is not None:
            text: List[int] = self.tokenizer(text, add_special_tokens=False)['input_ids']
        text: Tensor = torch.tensor(text, dtype=torch.long)
        return video_id, category, text, torch.tensor(highlight_start, dtype=torch.long)
