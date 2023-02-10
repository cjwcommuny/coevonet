from statistics import mean
from typing import List, Dict, Tuple

import pytorch_lightning as pl
import torch
from functionalstream import Stream
from torchtools.modules import ZeroLayer

from torchtools.optimization import weight_decay_parameters_group

from gcn_ranker.metrics.iou import RankIoU, IoU
from gcn_ranker.modules.jre.main_module import MainJre
from gcn_ranker.modules.video_grounding_baselines.ABLR import ABLR

from gcn_ranker.modules.video_grounding_baselines.LNLV import Lnlv
from gcn_ranker.modules.video_grounding_baselines.sqan import LGVTITG
from gcn_ranker.modules.video_highlight_detection_baselines.AttFM import AttFMWrapper
from gcn_ranker.modules.video_highlight_detection_baselines.video2gif import Video2Gif

_models = {
    'MainJre': MainJre,
    'Lnlv': Lnlv,
    'AttFM': AttFMWrapper,
    'ABLR': ABLR,
    'random': ZeroLayer,
    'video2gif': Video2Gif,
    'sqan': LGVTITG
}

class TaobaoBaseRunner(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.optimization_config = config['optimization']
        self.scheduler_config = config['scheduler']
        self.train_dataloader_config = config['train_dataloader']
        self.val_dataloader_config = config['val_dataloader']
        #
        self.config = config
        self.iou_settings: List[List[int, float]] = config['metrics']['iou']
        self.window_size = config['model_config']['window_size']
        #
        self.train_dataset_config = config['train_dataset']
        self.val_dataset_config = config['val_dataset']
        self.loader_config = config['loader']
        self.tokenizer = config['tokenizer']
        #
        self.seed = config['seed']
        #
        self.model = _models[config['model_type']](**config['model_config'])
        self.train_dataset = None
        self.val_dataset = None

    def forward(self, batch: tuple, batch_idx: int, mode: str):
        raise NotImplementedError()

    def running_step(self, batch: tuple, batch_idx: int, mode: str):
        (video_id, category, video_features, text_indexes, highlight_start), category_id = batch
        loss, window_scores, window_starts = self(batch, batch_idx, mode)
        #
        window_starts_sorted = window_starts[torch.argsort(window_scores, descending=True)]
        window_positions_sorted = torch.stack(
            (window_starts_sorted, window_starts_sorted + self.window_size),
            dim=1
        ).detach().cpu().numpy()
        ground_truth_window = torch.stack(
            (highlight_start, highlight_start + self.window_size)
        ).detach().cpu().numpy()
        recall_IoUs: Dict[str, float] = {
            f'R@{top_n},IoU={threshold}': RankIoU(
                top_n,
                threshold,
                window_positions_sorted,
                ground_truth_window
            )
            for top_n, threshold in self.iou_settings
        }
        iou = IoU(window_positions_sorted[0].reshape(1,2), ground_truth_window.reshape(1,2)).item()
        return {
            'loss': loss,
            'video_id': video_id,
            'category': category,
            'iou': {
                'iou': iou,
                **recall_IoUs
            },
            'recall_IoUs': recall_IoUs,
            'log': {
                f'loss/{mode}_batch': loss.item(),
            }
        }

    @staticmethod
    def avg_ious(ious: List[dict]) -> dict:
        names = ious[0].keys()
        return {
            name: mean(iou[name] for iou in ious)
            for name in names
        }

    @staticmethod
    def epoch_end(epoch: int, outputs: List[dict], mode: str):
        mean_loss = mean(out['loss'].item() for out in outputs)
        #
        ious_with_category: List[Tuple[dict, str]] = [(out['iou'], out['category']) for out in outputs]
        category_to_iou: Dict[str, Dict[str, float]] = Stream(ious_with_category)\
            .sorted(key=lambda x: x[1])\
            .groupby(key=lambda x: x[1])\
            .map(lambda category, iterator:
                (
                    category,
                    TaobaoBaseRunner.avg_ious([x[0] for x in list(iterator)])
                )
            )\
            .to_dict()
        ious_with_categories: Dict[str, float] = {
            f'{mode}_iou_{category}/{name}': value
            for category, ious in category_to_iou.items()
            for name, value in ious.items()
        }
        #
        ious: List[dict] = [out['iou'] for out in outputs]
        ious_avg = {f'{mode}_iou_avg/{name}': value for name, value in TaobaoBaseRunner.avg_ious(ious).items()}
        return {
            'log': {
                'step': epoch,
                f'loss/{mode}_epoch': mean_loss,
                **ious_avg,
                **ious_with_categories
            }
        }

    def training_step(self, batch: tuple, batch_idx: int):
        return self.running_step(batch, batch_idx, mode='train')

    @torch.no_grad()
    def validation_step(self, batch: tuple, batch_idx: int):
        return self.running_step(batch, batch_idx, mode='val')

    def training_epoch_end(self, outputs: List[dict]):
        return self.epoch_end(self.current_epoch, outputs, mode='train')

    def validation_epoch_end(self, outputs: List[dict]):
        return self.epoch_end(self.current_epoch, outputs, mode='val')

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def test_epoch_end(self, *args, **kwargs):
        return self.validation_epoch_end(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            weight_decay_parameters_group(self.model, self.optimization_config['weight_decay']),
            lr=self.optimization_config['lr']
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.scheduler_config)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def prepare_data(self):
        raise NotImplementedError()

    def train_dataloader(self):
        raise NotImplementedError()

    def val_dataloader(self):
        raise NotImplementedError()

    def test_dataloader(self):
        return self.val_dataloader()