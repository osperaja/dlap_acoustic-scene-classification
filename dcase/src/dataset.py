import torch
import pandas as pd
import soundfile as sf
import numpy as np
from typing import Literal, Dict, Union
from scipy.signal import resample_poly
from fractions import Fraction
import os
from torchaudio.transforms import MelSpectrogram


class AcousticScenesDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_name: Literal['train', 'val', 'test'],
            sample_rate: int = 44100,
            mono: bool = True,
            # base_data_path: str = '/data/baproj/dlap',
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
    ):
        super(AcousticScenesDataset, self).__init__()

        # initialize attributes
        self.data_path = os.path.join(base_data_path, dataset_name)
        self.sample_rate = sample_rate
        self.mono = mono
        self.multi_stream = multi_stream
        self.stream_cache_dir = stream_cache_dir
        self.resample_cache_dir = resample_cache_dir
        self.precompute_mel = precompute_mel
        self.mel_cache_dir = mel_cache_dir
        self.mel_config = mel_config or {}
        self.dataset_name = dataset_name
        self.normalize_audio = normalize_audio
        self.input_stream = input_stream
        self.input_channels = input_channels

        if multi_stream:
            from preprocessing import MultiStreamPreprocessor
            self.preprocessor = MultiStreamPreprocessor(
                sample_rate,
                cache_dir=stream_cache_dir or os.path.join(base_data_path, "preprocessed_features")
            )
            if mono:
                raise ValueError("multi_stream requires mono=False")
        if self.precompute_mel:
            if not multi_stream:
                raise ValueError("precompute_mel requires multi_stream=True")
            self.mel_cache_dir = self.mel_cache_dir or os.path.join(base_data_path, "precomputed_mels")
            self.mel_transform = MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.mel_config.get('n_fft', 2048),
                hop_length=self.mel_config.get('hop_length', None),
                f_min=self.mel_config.get('f_min', 0.0),
                f_max=self.mel_config.get('f_max', 22050.0),
                n_mels=self.mel_config.get('n_mels', 40),
            )
            tag_parts = [
                f"sr{self.sample_rate}",
                f"nfft{self.mel_config.get('n_fft', 2048)}",
                f"hop{self.mel_config.get('hop_length', 'def')}",
                f"fmin{self.mel_config.get('f_min', 0.0)}",
                f"fmax{self.mel_config.get('f_max', 22050.0)}",
                f"nm{self.mel_config.get('n_mels', 40)}",
            ]
            self._mel_cache_tag = "_".join(tag_parts)

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

        if self.normalize_audio:
            audio_data = audio_data / (np.abs(audio_data).max() + 1e-8)

        # resample if needed
        if self.sample_rate is not None and self.sample_rate != audio_sample_rate:
            cached_audio = None
            if self.resample_cache_dir:
                key = os.path.splitext(os.path.basename(example['audio_path']))[0]
                cache_path = os.path.join(
                    self.resample_cache_dir,
                    f"{self.dataset_name}_{key}_sr{self.sample_rate}.npy"
                )
                if os.path.exists(cache_path):
                    cached_audio = np.load(cache_path)
            if cached_audio is not None:
                audio_data = cached_audio
            else:
                sr_ratio = Fraction(self.sample_rate, audio_sample_rate)
                audio_data = resample_poly(
                    audio_data, up=sr_ratio.numerator, down=sr_ratio.denominator, axis=0
                )
                if self.resample_cache_dir:
                    os.makedirs(self.resample_cache_dir, exist_ok=True)
                    try:
                        np.save(cache_path, audio_data)
                    except Exception:
                        pass

        cache_key = f"{self.dataset_name}_{os.path.splitext(os.path.basename(example['audio_path']))[0]}"

        # Extract common values
        class_label = torch.tensor(example['class_label'])
        filename = example['audio_path']

        if self.multi_stream:
            # Convert to tensor and transpose: (SAMPLES, CHANNEL) -> (CHANNEL, SAMPLES)
            audio_tensor = torch.from_numpy(audio_data.T).float()  # (2, TIME)

            streams = self.preprocessor.process(audio_tensor, cache_key)

            if self.input_stream is not None:
                return {
                    'audio_data': streams[self.input_stream],  # (1, TIME)
                    'class_label': class_label,
                    'filename': filename,
                }

            if len(self.input_channels) == 2:
                return {
                    'audio_ch1': streams[self.input_channels[0]],
                    'audio_ch2': streams[self.input_channels[1]],
                    'class_label': class_label,
                    'filename': filename,
                }

            if len(self.input_channels) > 2:
                return {
                    'audio_ch1': streams[self.input_channels[0]],  # Left
                    'audio_ch2': streams[self.input_channels[1]],  # Right
                    'audio_ch3': streams[self.input_channels[2]],  # Mid
                    'audio_ch4': streams[self.input_channels[3]],  # Side
                    'audio_ch5': streams[self.input_channels[4]],  # Harmonic
                    'audio_ch6': streams[self.input_channels[5]],  # Percussive
                    'class_label': class_label,
                    'filename': filename,
                }

            # Full streams for ensemble - handle precomputed mels
            if self.precompute_mel:
                mel_cache_path = os.path.join(
                    self.mel_cache_dir,
                    f"{self.dataset_name}_{cache_key}_{self._mel_cache_tag}.pt"
                )
                if os.path.exists(mel_cache_path):
                    mels = torch.load(mel_cache_path, weights_only=True)
                else:
                    mels = {}
                    for stream_name, stream_audio in streams.items():
                        with torch.no_grad():
                            mel = torch.log(self.mel_transform(stream_audio) + 1e-6)
                        mels[stream_name] = mel
                    os.makedirs(self.mel_cache_dir, exist_ok=True)
                    try:
                        torch.save(mels, mel_cache_path)
                    except Exception:
                        pass
                return {
                    'mels': mels,
                    'class_label': class_label,
                    'filename': filename,
                }

            return {
                'streams': streams,
                'class_label': class_label,
                'filename': filename,
            }
        else:
            if self.mono:
                audio_data = audio_data.mean(axis=-1, keepdims=True)
            example['audio_data'] = torch.from_numpy(audio_data.T).float()
            example['class_label'] = class_label
            return example
