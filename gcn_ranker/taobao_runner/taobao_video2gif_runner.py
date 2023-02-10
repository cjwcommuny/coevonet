import torch
from torchtools.random import choice

from gcn_ranker.modules.video_highlight_detection_baselines.video2gif import HuberLoss
from gcn_ranker.taobao_runner.taobao_window_cls_runner import TabaoWindowClsRunner


class TaobaoVideo2gifRunner(TabaoWindowClsRunner):
    def __init__(self, config: dict):
        super().__init__(config)
        self.huber_loss = HuberLoss(config['loss']['delta'])

    def forward(self, batch: tuple, batch_idx: int, mode: str):
        (video_id, category, video_features, text_indexes, highlight_start), category_id = batch
        window_scores = self.model(video_features)
        window_starts = torch.arange(window_scores.shape[0])\
            .to(window_scores.device)
        highlight_score = window_scores[highlight_start]
        non_highlight_start = choice(low=0, high=window_scores.shape[0] - 1, num=1, device=window_scores.device).item()
        if non_highlight_start >= highlight_start:
            non_highlight_start += 1
        non_highlight_score = window_scores[non_highlight_start]
        loss = self.huber_loss(highlight_score, non_highlight_score)
        return loss, window_scores, window_starts
