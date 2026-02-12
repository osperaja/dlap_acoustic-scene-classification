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
            sample_rate: int = 44100,
            batch_size: int = 32,
            n_workers: int = 4,
            mono: bool = True,
            base_data_path: str = './data/dcase',
            normalize_audio: bool = False,
            multi_stream: bool = False,
            stream_cache_dir: str = None,
            resample_cache_dir: str = None,
            precompute_mel: bool = False,
            mel_cache_dir: str = None,
            mel_config: dict = None,
            input_stream: str = None,
            input_channels: list = None,
            shared_mel: bool = False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.mono = mono
        self.base_data_path = base_data_path
        self.normalize_audio = normalize_audio
        self.multi_stream = multi_stream
        self.stream_cache_dir = stream_cache_dir
        self.resample_cache_dir = resample_cache_dir
        self.precompute_mel = precompute_mel
        self.mel_cache_dir = mel_cache_dir
        self.mel_config = mel_config
        self.input_stream = input_stream
        self.input_channels = input_channels

        self._dataset_kwargs = dict(
            multi_stream=multi_stream,
            base_data_path=base_data_path,
            sample_rate=sample_rate,
            mono=mono,
            stream_cache_dir=stream_cache_dir,
            resample_cache_dir=resample_cache_dir,
            precompute_mel=precompute_mel,
            mel_cache_dir=mel_cache_dir,
            mel_config=mel_config,
            normalize_audio=normalize_audio,
            input_stream=input_stream,
            input_channels=input_channels,
        )

        # initialize datasets
        self.train_dataset = AcousticScenesDataset(
            dataset_name='train',
            **self._dataset_kwargs
        )

        self.val_dataset = AcousticScenesDataset(
            dataset_name='val',
            **self._dataset_kwargs
        )
        self.test_dataset = AcousticScenesDataset(
            dataset_name='test',
            multi_stream=multi_stream,
            base_data_path=base_data_path,
            sample_rate=sample_rate,
            mono=mono,
            normalize_audio=normalize_audio,
            input_stream=input_stream,
            input_channels=input_channels,
        )


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
