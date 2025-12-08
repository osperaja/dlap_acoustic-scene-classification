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
            n_label: int = 15,
            lstm_hidden=64,
            lstm_layers=1,
            cnn_dropout:float= 0.2,
            lstm_dropout:float= 0.5,
    ):
        super(BaselineModel, self).__init__()

        # mel spectrogram as feature transform
        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels
        )

        # nn architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.MaxPool2d((2, 1), stride=(2,1)),

            nn.Dropout(cnn_dropout),
        )

        lstm_input = 64 * (n_mels // 4)

        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Output-Layer
        self.fc_dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(lstm_hidden * 2, n_label)

    def forward(
            self,
            audio_data: torch.Tensor,  # (BATCH, CHANNEL=1, TIME)
    ) -> torch.Tensor:  # (BATCH, FRAMES', LABEL)
        assert audio_data.shape[1] == 1, ('baseline is single-channel', audio_data.shape[1])

        spec = torch.log(self.mel_transform(audio_data) + 1e-6)

        features = self.cnn(spec)

        B, C, F, T = features.shape
        features = features.permute(0, 3, 1, 2).reshape(B, T, C * F)

        features, _ = self.lstm(features)
        logits= self.output_layer(features)
        return logits