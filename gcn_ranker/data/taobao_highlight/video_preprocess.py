import cv2
from datascience.opencv import VideoCaptureIter, get_fps
from functionalstream.stream import OneOffStream
import os

def preprocess_video(video: cv2.VideoCapture, second_interval: float, save_dir: str):
    frame_interval = int(get_fps(video) * second_interval)
    os.makedirs(save_dir, exist_ok=True)
    video_iter = VideoCaptureIter(video)
    OneOffStream(video_iter)\
        .enumerate()\
        .filter(lambda idx, frame: idx % frame_interval == 0)\
        .foreach(lambda idx, frame: cv2.imwrite(os.path.join(save_dir, f'{idx}.jpg'), frame))\
        .consume()
    print(f'finish {save_dir}')
