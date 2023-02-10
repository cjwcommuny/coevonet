import torch
from torch import Tensor

from gcn_ranker.metrics.smooth_l1 import smooth_l1_norm
from gcn_ranker.modules.common import index2mask_1d
from gcn_ranker.taobao_runner.taobao_window_cls_runner import TabaoWindowClsRunner

class TaobaoWindowRegressionRunner(TabaoWindowClsRunner):
    def forward(self, batch: tuple, batch_idx: int, mode: str):
        (video_id, category, video_features, text_indexes, highlight_start), category_id = batch
        position_normalized = self.model(video_features, text_indexes)
        T = video_features.shape[0]
        num_windows = T - self.model.window_size + 1
        device = video_features.device
        window_starts = torch.arange(start=0, end=num_windows, step=1).to(device)
        window_start = round(position_normalized[0].item() * (len(window_starts) - 1))
        window_scores_pseudo = index2mask_1d(
            torch.tensor([window_start], dtype=torch.long, device=device),
            num_windows
        )
        highlight_start_normalized= highlight_start.float() / num_windows
        loss = smooth_l1_norm(position_normalized[0] - highlight_start_normalized)
        return loss, window_scores_pseudo, window_starts
    