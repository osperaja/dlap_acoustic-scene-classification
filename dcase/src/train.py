import argparse

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint, EarlyStopping
import yaml
from torch.multiprocessing import set_start_method
try:
    from .datamodule import AcousticScenesDatamodule as DM
    from .models import BaselineModel, LinSeqModel, CNNModel, EnsembleCNNModel
    from .baseline_experiment import BaselineExperiment as BLExp
    from .cnn_experiment import CNNExperiment as CNNExp
except ImportError:
    from datamodule import AcousticScenesDatamodule as DM
    from models import BaselineModel, LinSeqModel, CNNModel, EnsembleCNNModel
    from baseline_experiment import BaselineExperiment as BLExp
    from cnn_experiment import CNNExperiment as CNNExp

MODEL_REGISTRY = {
    'BaselineModel': BaselineModel,
    'LinSeqModel': LinSeqModel,
    'CNNModel': CNNModel,
    "EnsembleCNNModel": EnsembleCNNModel,
}

EXPERIMENT_REGISTRY = {
    'BaselineModel': BLExp,
    'LinSeqModel': BLExp,
    'CNNModel': CNNExp,
    'EnsembleCNNModel': CNNExp,
}


class EarlyStoppingFromEpoch(EarlyStopping):
    def __init__(self, start_epoch: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch

    def on_validation_end(self, trainer, pl_module, *args, **kwargs):
        if trainer.current_epoch >= self.start_epoch:
            super().on_validation_end(trainer, pl_module, *args, **kwargs)


def setup_logging(tb_log_dir: str, exp_name: str, version_id: int = None):
    if version_id is None:
        tb_logger = pl_loggers.TensorBoardLogger(tb_log_dir, name=exp_name, log_graph=False)
        version_id = int((tb_logger.log_dir).split('_')[-1])
    else:
        tb_logger = pl_loggers.TensorBoardLogger(tb_log_dir, name=exp_name, log_graph=False, version=version_id)
    return tb_logger, version_id


def get_trainer(devices, logger, max_epochs, strategy, accelerator, ckpt_dir):
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=3,
        mode='max',
        monitor="val/accuracy",
        auto_insert_metric_name=False,
        filename='epoch={epoch}-val_acc={val/accuracy:.2f}'
    )
    early_stop = EarlyStoppingFromEpoch(
        start_epoch=0,
        monitor="val/accuracy",
        patience=10,
        mode="max",
        verbose=True
    )
    return pl.Trainer(
        enable_model_summary=True,
        logger=logger,
        devices=devices,
        max_epochs=max_epochs,
        strategy=strategy,
        accelerator=accelerator,
        callbacks=[checkpoint_callback, ModelSummary(max_depth=2), early_stop],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/baseline.yaml', help='Path to config YAML')
    args = parser.parse_args()

    set_start_method('spawn', force=True)

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    model_name = config.get('model', 'BaselineModel')
    ModelClass = MODEL_REGISTRY[model_name]
    ExpClass = EXPERIMENT_REGISTRY[model_name]

    # create model based on type
    if model_name == 'EnsembleCNNModel':
        model = ModelClass(base_model_config=config['network']['base_model_config'])
    else:
        config['network']['sample_rate'] = config['data']['sample_rate']
        config['network']['n_label'] = config['experiment']['n_label']
        model = ModelClass(**config['network'])

    dm = DM(**config['data'])
    exp = ExpClass(model=model, **config['experiment'])

    exp_name = model_name.lower()
    tb_logger, version = setup_logging(config['logging']['tb_log_dir'], exp_name)
    ckpt_dir = config['logging']['ckpt_dir'] + exp_name

    trainer = get_trainer(logger=tb_logger, **config['training'], ckpt_dir=ckpt_dir)
    trainer.fit(exp, dm)
