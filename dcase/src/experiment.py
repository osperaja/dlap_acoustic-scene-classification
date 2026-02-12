import pytorch_lightning as pl
import torch
from typing import Literal, Dict, Union


class AcousticScenesExperiment(pl.LightningModule):
    def __init__(
            self,
            learning_rate: float,
            max_epochs: int,
            optimizer_type: str = "sgd",
            use_scheduler: bool = True,
    ):
        super(AcousticScenesExperiment, self).__init__()

        # init attributes
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.optimizer_type = optimizer_type
        self.use_scheduler = use_scheduler

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
        opt_type = (self.optimizer_type or "sgd").lower()
        if opt_type == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif opt_type == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        elif opt_type == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=1e-4,
                momentum=0.9,
                nesterov=True
            )
        else:
            raise ValueError(f"Unsupported optimizer_type: {self.optimizer_type}")

        print(f"\033[91mUsing {opt_type} optimizer with learning rate {self.learning_rate}")
        print(f"\033[91mUsing scheduler: {self.use_scheduler}\033[0m")

        if not self.use_scheduler:
            return optimizer

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }
