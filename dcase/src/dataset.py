import torch
import pandas as pd
import soundfile as sf
import numpy as np
from typing import Literal, Dict, Union
from scipy.signal import resample_poly
from fractions import Fraction
import os


class AcousticScenesDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_name: Literal['train', 'val', 'test'],
            sample_rate: int = 44100,
            mono: bool = True,
            base_data_path: str = './data/dcase',
            multi_stream: bool = False,
    ):
        super(AcousticScenesDataset, self).__init__()

        # initialize attributes
        self.data_path = os.path.join(base_data_path, dataset_name)
        self.sample_rate = sample_rate
        self.mono = mono
        self.multi_stream = multi_stream

        if multi_stream:
            from preprocessing import MultiStreamPreprocessor
            self.preprocessor = MultiStreamPreprocessor(
                sample_rate,
                cache_dir=os.path.join(base_data_path, "preprocessed_features")
            )
            if mono:
                raise ValueError("multi_stream requires mono=False")

        if dataset_name == 'test':
            self.meta_df = pd.read_csv(
                self.data_path + '/meta_blind.txt', delimiter='\t', header=None,
                names=['audio_path', 'scene_name', 'scene_id']
            )
        # load meta
        else:
            self.meta_df = pd.read_csv(
                self.data_path + '/meta.txt', delimiter='\t', header=None,
                names=['audio_path', 'scene_name', 'scene_id']
            )

        # remove broken samples in development set
        if dataset_name == 'train':
            error_df = pd.read_csv(
                self.data_path + '/error.txt', delimiter='\t', header=None, usecols=[0], names=['audio_path']
            )
            error_mask = self.meta_df['audio_path'].isin(error_df['audio_path'])
            self.meta_df = self.meta_df[~error_mask].reset_index(drop=True)

        # create class labels
        scene_names = sorted(self.meta_df['scene_name'].unique())
        class_affiliation = {
            scene_name: class_idx for class_idx, scene_name in enumerate(scene_names)
        }
        self.meta_df['class_label'] = self.meta_df['scene_name'].map(class_affiliation)

    def __len__(self) -> int:
        return len(self.meta_df)

    def __getitem__(self, idx) -> Dict[str, Union[str, int, torch.Tensor]]:
        # load meta dict
        example = self.meta_df.iloc[idx].to_dict()

        # load audio
        audio_data, audio_sample_rate = sf.read(
            self.data_path + '/' + example['audio_path'], dtype=np.float32
        )  # (SAMPLES, CHANNEL)

        # resample if needed
        if self.sample_rate is not None and self.sample_rate != audio_sample_rate:
            sr_ratio = Fraction(self.sample_rate, audio_sample_rate)
            audio_data = resample_poly(
                audio_data, up=sr_ratio.numerator, down=sr_ratio.denominator, axis=0
            )

        if self.multi_stream:
            key = os.path.splitext(os.path.basename(example['audio_path']))[0]
            streams = self.preprocessor.process(
                torch.from_numpy(audio_data.T).float(),
                cache_key=key
            )
            example['streams'] = streams
        else:
            if self.mono:
                audio_data = audio_data.mean(axis=-1, keepdims=True)
            example['audio_data'] = torch.from_numpy(audio_data.T).float()

        example['class_label'] = torch.tensor(example['class_label'])
        return example
