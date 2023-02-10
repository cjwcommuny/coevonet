from cached_property import cached_property
from functionalstream import Stream

from gcn_ranker.taobao_runner.taobao_frames_runner import TaobaoFramesRunner
import torch

from gcn_ranker.utils import DiscreteDistribution


class TaobaoRandomRunner(TaobaoFramesRunner):
    def forward(self, batch: tuple, batch_idx: int, mode: str):
        (video_id, category, video_frames, text_indexes, highlight_start), category_id = batch
        pseudo_loss = torch.tensor(0.0, device=video_frames.device, requires_grad=True)
        #
        T = video_frames.shape[0]
        num_windows = T - self.window_size + 1
        window_starts = torch.arange(start=0, end=num_windows, step=1).to(video_frames.device)
        window_scores = torch.zeros(size=window_starts.shape, device=window_starts.device)
        predict_indice = 0 if self.config['model_config']['type'] == 'head' else int(self.dataset_highlight_start_distribution.draw() * num_windows)
        window_scores[predict_indice] = 1
        return pseudo_loss, window_scores, window_starts

    @cached_property
    def dataset_highlight_start_distribution(self):
        dataset = self.train_dataset
        distribution = Stream(dataset)\
            .map(lambda data, _: (len(data[2]), data[4].item()))\
            .map(lambda T, highlight_start: highlight_start / (T - self.window_size + 1))\
            .to_list()
        return DiscreteDistribution(distribution, ndigits=1)
