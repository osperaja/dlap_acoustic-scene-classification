import pytorch_lightning as pl
import torch
from typing import Literal, Dict, Union


class AcousticScenesExperiment(pl.LightningModule):
    def __init__(
            self,
            learning_rate: float,
        ):
        super(AcousticScenesExperiment, self).__init__()

        # init attributes
        self.learning_rate = learning_rate
    
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

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
   