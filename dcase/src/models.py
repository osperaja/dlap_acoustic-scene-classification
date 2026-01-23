import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram


class BaselineModel(torch.nn.Module):
    """DCASE baseline MLP model.
    - 40 mel bands, 0-22050 Hz
    - 5-frame context (feature vector length = 200)
    - 2 hidden layers × 50 units, 20% dropout
    - Softmax output for 15 classes
    """

    def __init__(
            self,
            sample_rate: int = 44100,
            n_fft: int = 2048,
            hop_length: int = 882,
            f_min: float = 0.0,
            f_max: float = 22050.0,
            n_mels: int = 40,
            n_context: int = 5,
            n_hidden: int = 50,
            n_hidden_layers: int = 2,
            dropout: float = 0.2,
            n_label: int = 15,
    ):
        super(BaselineModel, self).__init__()

        self.n_context = n_context
        self.n_mels = n_mels

        # mel spectrogram as feature transform
        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
        )

        input_size = n_mels * n_context

        layers = [nn.Linear(input_size, n_hidden), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(n_hidden, n_label))

        self.network = nn.Sequential(*layers)

    def forward(
            self,
            audio_data: torch.Tensor,  # (BATCH, CHANNEL=1, TIME)
    ) -> torch.Tensor:  # (BATCH, FRAMES', LABEL)

        # (BATCH, 1, N_MELS, FRAMES)
        spec = torch.log(self.mel_transform(audio_data) + 1e-6)
        spec = spec.squeeze(1)  # (BATCH, N_MELS, FRAMES)

        B, F, T = spec.shape
        pad = self.n_context // 2

        spec_padded = nn.functional.pad(spec, (pad, pad), mode='replicate')

        spec_unfolded = spec_padded.unfold(dimension=2, size=self.n_context, step=1)

        features = spec_unfolded.permute(0, 2, 1, 3).reshape(B, T, -1)

        logits = self.network(features)
        return logits


class LinSeqModel(torch.nn.Module):
    """
    Extended linear sequential model with optional SpecAugment.
    """

    def __init__(
            self,
            sample_rate: int = 44100,
            n_fft: int = 2048,
            f_min: float = 0.0,
            f_max: float = 22050.0,
            n_mels: int = 40,
            dropout: float = 0.2,
            n_label: int = 15,
            n_hidden_feats: int = 50,
            n_hidden_layer: int = 4,
            spec_augment: bool = True,
            freq_mask_param: int = 10,
            time_mask_param: int = 20,
            n_freq_masks: int = 2,
            n_time_masks: int = 2,
    ):
        super(LinSeqModel, self).__init__()

        self.spec_augment = spec_augment

        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
        )

        # SpecAugment: frequency and time masking
        if spec_augment:
            from torchaudio.transforms import FrequencyMasking, TimeMasking
            self.freq_masks = nn.ModuleList([
                FrequencyMasking(freq_mask_param) for _ in range(n_freq_masks)
            ])
            self.time_masks = nn.ModuleList([
                TimeMasking(time_mask_param) for _ in range(n_time_masks)
            ])

        layers = [nn.Linear(n_mels, n_hidden_feats), nn.GELU(), nn.Dropout(dropout)]

        for _ in range(n_hidden_layer - 1):
            layers.append(nn.Linear(n_hidden_feats, n_hidden_feats))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(n_hidden_feats, n_label))

        self.network = nn.Sequential(*layers)

    def forward(
            self,
            audio_data: torch.Tensor,  # (BATCH, CHANNEL=1, TIME)
    ) -> torch.Tensor:  # (BATCH, FRAMES', LABEL)
        assert audio_data.shape[1] == 1, ('baseline is single-channel', audio_data.shape[1])

        features = torch.log(self.mel_transform(audio_data) + 1e-6)  # (B, 1, n_mels, T)

        # SpecAugment only during training
        if self.spec_augment and self.training:
            for freq_mask in self.freq_masks:
                features = freq_mask(features)
            for time_mask in self.time_masks:
                features = time_mask(features)

        aggregated_features = features.squeeze(1).transpose(1, 2)  # (BATCH, FRAMES', FEAT)

        logits = self.network(aggregated_features)

        return logits

class CNNModel(torch.nn.Module):
    """
    CNN for spectrogram classification with SpecAugment and Mixup.
    """

    def __init__(
            self,
            sample_rate: int = 44100,
            n_fft: int = 2048,
            f_min: float = 0.0,
            f_max: float = 22050.0,
            n_mels: int = 40,
            dropout: float = 0.3,
            n_label: int = 15,
            spec_augment: bool = True,
            freq_mask_param: int = 15,
            time_mask_param: int = 20,
            n_freq_masks: int = 2,
            n_time_masks: int = 2,
            mixup_alpha: float = 0.2,
            use_mixup: bool = True,
            conv_channels: list = None,  # e.g. [64, 128, 256]
            classifier_hidden: int = 128,
    ):
        super(CNNModel, self).__init__()

        if conv_channels is None:
            conv_channels = [64, 128, 256]

        self.spec_augment = spec_augment
        self.n_mels = n_mels
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
        )

        if spec_augment:
            from torchaudio.transforms import FrequencyMasking, TimeMasking
            self.freq_masks = nn.ModuleList([
                FrequencyMasking(freq_mask_param) for _ in range(n_freq_masks)
            ])
            self.time_masks = nn.ModuleList([
                TimeMasking(time_mask_param) for _ in range(n_time_masks)
            ])

        # Build CNN blocks dynamically
        blocks = []
        in_channels = 1
        for out_channels in conv_channels:
            blocks.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout),
            ])
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(conv_channels[-1], classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, n_label),
        )

    def mixup_data(self, x, y):
        """Apply mixup to batch. Returns mixed inputs and label pairs with lambda."""
        lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def forward(
            self,
            audio_data: torch.Tensor,  # (BATCH, 1, TIME)
            labels: torch.Tensor = None,  # (BATCH,) for mixup
    ) -> dict:
        # (BATCH, 1, n_mels, FRAMES)
        features = torch.log(self.mel_transform(audio_data) + 1e-6)

        if self.spec_augment and self.training:
            for freq_mask in self.freq_masks:
                features = freq_mask(features)
            for time_mask in self.time_masks:
                features = time_mask(features)

        # Apply mixup during training
        y_a, y_b, lam = None, None, None
        if self.use_mixup and self.training and labels is not None:
            features, y_a, y_b, lam = self.mixup_data(features, labels)

        x = self.conv_blocks(features)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)

        if self.training and y_a is not None:
            return {"logits": logits, "y_a": y_a, "y_b": y_b, "lam": lam}
        return {"logits": logits}