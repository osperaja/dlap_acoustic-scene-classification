from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
try:
    from .dataset import AcousticScenesDataset
except ImportError:
    from dataset import AcousticScenesDataset
from typing import List, Dict, Union

class AcousticScenesDatamodule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size: int = 8,
            n_workers: int = 16,
            sample_rate: int = 44100,
            mono: bool = False,
            base_data_path: str = './data/dcase',
            multi_stream: bool = True,
    ):
        super(AcousticScenesDatamodule, self).__init__()

        # initialize attributes
        self.batch_size = batch_size
        self.n_workers = n_workers

        # initialize datasets
        self.train_dataset = AcousticScenesDataset(
            dataset_name='train',
            multi_stream=multi_stream,
            base_data_path=base_data_path,
            sample_rate=sample_rate,
            mono=mono,
        )

        self.val_dataset = AcousticScenesDataset(
            dataset_name='val',
            multi_stream=multi_stream,
            base_data_path=base_data_path,
            sample_rate=sample_rate,
            mono=mono,
        )
        # self.test_dataset = AcousticScenesDataset(
        #     dataset_name='test',
        #     multi_stream=multi_stream,
        #     base_data_path=base_data_path,
        #     sample_rate=sample_rate,
        #     mono=mono,
        # )

        if self.train_dataset.multi_stream:
            for idx in tqdm(range(len(self.train_dataset)), desc="Precompute train streams"):
                _ = self.train_dataset[idx]

        if self.val_dataset.multi_stream:
            for idx in tqdm(range(len(self.val_dataset)), desc="Precompute val streams"):
                _ = self.val_dataset[idx]

        # if self.val_dataset.multi_stream:
        #     for idx in tqdm(range(len(self.test_dataset)), desc="Precompute val streams"):
        #         _ = self.val_dataset[idx]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if self.n_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
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
        (
            {k: torch.stack([sample[key][k] for sample in batch], dim=0) for k in value.keys()}
            if isinstance(value, dict) else
            [sample[key] for sample in batch]
        )
        for key, value in batch[0].items()
    }
