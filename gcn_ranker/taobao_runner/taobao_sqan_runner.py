from gcn_ranker.modules.common import index2mask_1d
from gcn_ranker.taobao_runner import TabaoWindowClsRunner
import torch

class TaobaoSqanRunner(TabaoWindowClsRunner):
    def forward(self, batch: tuple, batch_idx: int, mode: str):
        (video_id, category, video_features, text_indexes, highlight_start), category_id = batch
        position_normalized, loss, reg_loss, tag_loss, dqa_loss = self.model(video_features, text_indexes, highlight_start)
        #
        T = video_features.shape[0]
        num_windows = T - self.model.window_size + 1
        device = video_features.device
        window_starts = torch.arange(start=0, end=num_windows, step=1).to(device)
        window_start = round(torch.clamp(position_normalized[0], 0, 1).item() * (len(window_starts) - 1))
        assert 0 <= window_start < len(window_starts), f'{video_id=}, {window_start=}, {len(window_starts)=}, {position_normalized=}'
        window_scores_pseudo = index2mask_1d(
            torch.tensor([window_start], dtype=torch.long, device=device),
            num_windows
        )
        return loss, window_scores_pseudo, window_starts
