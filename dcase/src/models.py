import torch
from torchaudio.transforms import MelSpectrogram
from typing import Literal, List


class BaselineModel(torch.nn.Module):
    def __init__(
            self,
            sample_rate: int = 44100,
            n_fft: int = 2048,
            f_min: float = 0.0,
            f_max: float = 22050.0,
            n_mels: int = 40,
            dropout: float = 0.2,
            n_aggregate: int = 5,
            n_label: int = 15,
            n_hidden_feats: int = 50,
            n_hidden_layer: int = 2,
        ):
        super(BaselineModel, self).__init__()

        # init attributes
        self.n_aggregate = n_aggregate

        # mel spectrogram as feature transform
        self.mel_transform = MelSpectrogram(
            ???
        )

        # nn architecture
        self.network = ???

    def forward(
            self, 
            audio_data: torch.Tensor,  # (BATCH, CHANNEL=1, TIME)
        ) -> torch.Tensor:  # (BATCH, FRAMES', LABEL)
        assert audio_data.shape[1] == 1, ('baseline is single-channel', audio_data.shape[1])

        # compute log-mel spectrogram as feature transform
        features = ???  # (BATCH, FEAT, FRAMES)

        # aggregate features
        aggregated_features = ???  # (BATCH, FEAT, FRAMES')
        
        # forward network
        logits = ???  # (BATCH, FRAMES', LABEL)
        
        return logits


