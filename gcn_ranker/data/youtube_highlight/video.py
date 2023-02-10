import os
from pathlib import Path

from cv2.cv2 import VideoCapture
from datascience.opencv import VideoCaptureIter
from functionalstream import Stream
from functionalstream.stream import OneOffStream
from torch.utils.data import Dataset

from gcn_ranker.data.youtube_highlight.youtube_highlight import YoutubeHighlight


class YoutubeHighlightVideo(Dataset):
    def __init__(self, sample_interval_frame, *args, **kwargs):
        super().__init__()
        self.sample_interval_frame = sample_interval_frame
        self.dataset = YoutubeHighlight(*args, **kwargs, loader=self.loader)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

    def loader(self, video_dir: str):
        video_id = Path(video_dir).stem
        video_path = Stream(self.dataset.VIDEO_EXTENSIONS) \
            .map(lambda extension: os.path.join(video_dir, f'{video_id}{extension}')) \
            .filter(lambda path: os.path.exists(path)) \
            .to_list()[0]
        idxes_frames = self.extract_frames(VideoCapture(video_path), self.sample_interval_frame)
        # return VideoCapture(video_path), video_dir
        return idxes_frames, video_dir

    @staticmethod
    def extract_frames(
            video: VideoCapture,
            sample_interval_frame: int
    ) -> Stream:
        """
        :return frames.shape=(T, H, W, C), range=[0,1]
        """
        return OneOffStream(VideoCaptureIter(video)) \
            .enumerate() \
            .filter(lambda idx, frame: idx % sample_interval_frame == 0)