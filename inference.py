import argparse
from gcn_ranker.taobao_runner import RUNNING_TYPES
import pytorch_lightning as pl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runner_type')
    parser.add_argument('--checkpoint_path')
    parser.add_argument('--hparams_path')
    parser.add_argument('--gpus', type=int)
    args = parser.parse_args()
    #
    runner = RUNNING_TYPES[args.runner_type].load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        hparams_file=args.hparams_path,
        map_location=None
    )
    print(type(f'{runner.device=}'))
    trainer = pl.Trainer(gpus=[args.gpus])
    trainer.test(runner)
