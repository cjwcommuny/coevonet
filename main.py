import argparse
import importlib
from pathlib import Path

import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchtools.deterministic import set_seed

import pytorch_lightning as pl

from gcn_ranker.taobao_runner import RUNNING_TYPES

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()
    if Path(args.config).suffix in {'.yaml', '.yml'}:
        # load as yaml config
        config = yaml.load(open(args.config), 'r')
    else:
        # load as module
        config_mod = importlib.import_module(args.config)
        config = config_mod.config
    #
    set_seed(config['seed'])
    trainer = pl.Trainer(
        logger=TensorBoardLogger(**config['logger']),
        checkpoint_callback=ModelCheckpoint(**config['checkpoint_callback']),
        **config['trainer']
    )
    runner = RUNNING_TYPES[config['running_type']](config)
    trainer.fit(runner)
