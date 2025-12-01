import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram


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
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels
        )

        # nn architecture
        layers = [nn.Conv1d(n_mels, n_hidden_feats, kernel_size=1), nn.ReLU(), nn.Dropout(dropout)]

        for _ in range(n_hidden_layer - 1):
            layers.append(nn.Conv1d(n_hidden_feats, n_hidden_feats, kernel_size=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Conv1d(n_hidden_feats, n_label, kernel_size=1))

        self.network = nn.Sequential(*layers)

    def forward(
            self, 
            audio_data: torch.Tensor,  # (BATCH, CHANNEL=1, TIME)
        ) -> torch.Tensor:  # (BATCH, FRAMES', LABEL)
        assert audio_data.shape[1] == 1, ('baseline is single-channel', audio_data.shape[1])

        # compute log-mel spectrogram as feature transform
        features = torch.log(self.mel_transform(audio_data) + 1e-6)  # (BATCH, FEAT, FRAMES)

        # aggregate features
        aggregated_features = features.squeeze(1) # (BATCH, FEAT, FRAMES')
        
        # forward network
        logits = self.network(aggregated_features).transpose(1, 2)  # (BATCH, FRAMES', LABEL)
        
        return logits