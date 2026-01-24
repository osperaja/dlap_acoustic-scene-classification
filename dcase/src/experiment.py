import pytorch_lightning as pl
import torch
from typing import Literal, Dict, Union


class AcousticScenesExperiment(pl.LightningModule):
    def __init__(
            self,
            learning_rate: float,
            max_epochs: int,
    ):
        super(AcousticScenesExperiment, self).__init__()

        # init attributes
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def shared_step(
            self,
            batch: Dict[str, Union[torch.Tensor, str, int]],
            batch_idx: int,
            stage: Literal['train', 'val']
    ) -> torch.Tensor:
        raise NotImplementedError

    def training_step(
            self,
            batch: Dict[str, Union[torch.Tensor, str, int]],
            batch_idx: int,
    ) -> torch.Tensor:
        return self.shared_step(batch, batch_idx, stage='train')

    def validation_step(
            self,
            batch: Dict[str, Union[torch.Tensor, str, int]],
            batch_idx: int,
    ) -> torch.Tensor:
        return self.shared_step(batch, batch_idx, stage='val')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        }
