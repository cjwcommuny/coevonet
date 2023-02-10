import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer

from gcn_ranker.data.dataset_folder import DatasetFolderOfFolder, PredefinedDatasetFolder
from gcn_ranker.data.taobao_highlight.dataset import TaobaoFeatureLoader, FramesPackedLoader, is_valid_folder
from gcn_ranker.data.utils import RatioSubset, ratio_random_split
from gcn_ranker.metrics.pairwise import get_contrastive_pairs
from gcn_ranker.modules.common import index2mask_1d
from gcn_ranker.taobao_runner.taobao_base_runner import TaobaoBaseRunner
import torch.nn.functional as F

class TaobaoFramesRunner(TaobaoBaseRunner):
    def forward(self, batch: tuple, batch_idx: int, mode: str):
        (video_id, category, video_frames, text_indexes, highlight_start), category_id = batch
        window_scores, window_starts = self.model(video_frames, text_indexes)
        # assert window_step == 1s
        highlight_labels = index2mask_1d(highlight_start.view(-1), window_scores.shape[0])
        score_pairs = get_contrastive_pairs(window_scores, highlight_labels)
        pairwise_loss = F.margin_ranking_loss(
            input1=score_pairs[:,0],
            input2=score_pairs[:,1],
            target=torch.tensor([1], device=score_pairs.device),
            margin=1
        )
        return pairwise_loss, window_scores, window_starts

    def prepare_data(self):
        self.train_dataset = PredefinedDatasetFolder(
            **self.train_dataset_config,
            loader=FramesPackedLoader(
                **self.loader_config,
                tokenizer=AutoTokenizer.from_pretrained(self.tokenizer)
            )
        )
        self.val_dataset = PredefinedDatasetFolder(
            **self.val_dataset_config,
            loader=FramesPackedLoader(
                **self.loader_config,
                tokenizer=AutoTokenizer.from_pretrained(self.tokenizer)
            )
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=TaobaoFeatureLoader.collate_fn,
            **self.train_dataloader_config
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=TaobaoFeatureLoader.collate_fn,
            **self.val_dataloader_config
        )