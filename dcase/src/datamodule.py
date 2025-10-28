import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from dcase.src.dataset import AcousticScenesDataset
from typing import List, Dict, Union


class AcousticScenesDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 8,
        n_workers: int = 16,
        **ds_kwargs
        ):
        super(AcousticScenesDatamodule, self).__init__()

        # initialize attributes
        self.batch_size = batch_size
        self.n_workers = n_workers

        # initialize datasets
        self.train_dataset = AcousticScenesDataset(
            dataset_name='train',
            **ds_kwargs,
        )
        self.val_dataset = AcousticScenesDataset(
            dataset_name='val',
            **ds_kwargs,
        )
        self.test_dataset = AcousticScenesDataset(
            dataset_name='test',
            **ds_kwargs,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=True,
            collate_fn=collate_fn,
            persistent_workers=True if self.n_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=False,
            collate_fn=collate_fn, 
            persistent_workers=True if self.n_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            collate_fn=collate_fn, 
        )

def collate_fn(
    batch: List[Dict[str, Union[torch.Tensor, str, int]]]
    ) -> Dict[str, Union[torch.Tensor, str, int]]:
    return {
        key: torch.stack(
                [sample[key] for sample in batch], dim=0
            ) 
            if isinstance(value, torch.Tensor) else
            [
                sample[key] for sample in batch
            ]
         for key, value in batch[0].items()
    }

