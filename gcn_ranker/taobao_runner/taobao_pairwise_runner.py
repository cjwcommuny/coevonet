from gcn_ranker.modules.video_highlight_detection_baselines.video2gif import HuberLoss
from gcn_ranker.taobao_runner import TabaoWindowClsRunner
from torchtools.random import choice

class TaobaoPairwiseRunner(TabaoWindowClsRunner):
    def __init__(self, config: dict):
        super().__init__(config)
        self.huber_loss = HuberLoss(config['loss']['delta'])

    def forward(self, batch: tuple, batch_idx: int, mode: str):
        (video_id, category, video_features, text_indexes, highlight_start), category_id = batch
        window_scores, window_starts = self.model(video_features, text_indexes)
        highlight_score = window_scores[highlight_start]
        non_highlight_start = choice(low=0, high=window_scores.shape[0] - 1, num=1, device=window_scores.device).item()
        if non_highlight_start >= highlight_start:
            non_highlight_start += 1
        non_highlight_score = window_scores[non_highlight_start]
        loss = self.huber_loss(highlight_score, non_highlight_score)
        return loss, window_scores, window_starts
