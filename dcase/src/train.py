import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint, EarlyStopping
import yaml
from torch.multiprocessing import set_start_method
from dcase.src.datamodule import AcousticScenesDatamodule as DM

# experiment specific imports and settings
from dcase.src.models import BaselineModel as Model
from dcase.src.baseline_experiment import BaselineExperiment as Exp
CONFIG_PATH = 'config/baseline.yaml'
EXP_NAME = 'baseline'


def setup_logging(tb_log_dir: str, exp_name: str, version_id: int=None):
    if version_id is None:
        tb_logger = pl_loggers.TensorBoardLogger(tb_log_dir, name=exp_name, log_graph=False)
        version_id = int((tb_logger.log_dir).split('_')[-1])
    else:
        tb_logger = pl_loggers.TensorBoardLogger(tb_log_dir, name=exp_name, log_graph=False, version=version_id)

    return tb_logger, version_id


def get_trainer(devices, logger, max_epochs, strategy, accelerator, ckpt_dir):
    early_stopping_callback = EarlyStopping(
        monitor='val/accuracy',
        patience=5,
        mode='max',
        verbose=True
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=3,
        mode='max',
        monitor="val/accuracy",
        auto_insert_metric_name=False,
        filename='epoch={epoch}-val_acc={val/accuracy:.2f}'
    )
    return pl.Trainer(enable_model_summary=True,
        logger=logger,
        devices=devices,
        max_epochs=max_epochs,
        strategy = strategy,
        accelerator = accelerator,
        callbacks=[checkpoint_callback, early_stopping_callback, ModelSummary(max_depth=2)],
    )


if __name__== "__main__":
    # invoke multiprocessing
    set_start_method('spawn', force=True)

    # instantiate experiment
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
    config['network']['sample_rate'] = config['data']['sample_rate']
    config['network']['n_label'] = config['experiment']['n_label']
    dm = DM(**config['data'])
    model = Model(
        **config['network'], 
    )
    exp = Exp(
        model=model, 
        **config['experiment'],
    )

    # setup logging 
    tb_logger, version = setup_logging(config['logging']['tb_log_dir'], EXP_NAME)
    ckpt_dir = config['logging']['ckpt_dir'] + EXP_NAME

    # train
    trainer = get_trainer(logger=tb_logger, **config['training'], ckpt_dir=ckpt_dir)
    trainer.fit(exp, dm)
