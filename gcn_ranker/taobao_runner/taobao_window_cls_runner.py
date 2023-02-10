import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer

from gcn_ranker.data.dataset_folder import PredefinedDatasetFolder
from gcn_ranker.data.taobao_highlight.dataset import TaobaoFeatureLoader
from gcn_ranker.taobao_runner.taobao_base_runner import TaobaoBaseRunner
import torch


class TabaoWindowClsRunner(TaobaoBaseRunner):
    def forward(self, batch: tuple, batch_idx: int, mode: str):
        (video_id, category, video_features, text_indexes, highlight_start), category_id = batch
        if 'language_input' in self.config and not self.config['language_input']:
            text_indexes = torch.zeros_like(text_indexes)
        window_scores, window_starts = self.model(video_features, text_indexes)
        cls_loss = F.cross_entropy(window_scores.unsqueeze(0), highlight_start.view(-1))
        return cls_loss, window_scores, window_starts

    def prepare_data(self):
        self.train_dataset = PredefinedDatasetFolder(
            **self.train_dataset_config,
            loader=TaobaoFeatureLoader(
                **self.loader_config,
                tokenizer=AutoTokenizer.from_pretrained(self.tokenizer)
            )
        )
        self.val_dataset = PredefinedDatasetFolder(
            **self.val_dataset_config,
            loader=TaobaoFeatureLoader(
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