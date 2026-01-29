import numpy as np
import scipy
import torch
import torchaudio
from scipy.stats import skew
from sklearn.linear_model import LogisticRegression
from sympy.stats import kurtosis
from torch import nn
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm

from preprocessing import MultiStreamPreprocessor

from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin


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
        with torch.no_grad():
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

        with torch.no_grad():
            features = torch.log(self.mel_transform(audio_data) + 1e-6)  # (B, 1, n_mels, T)

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
            pooling=[3, 3],
            last_layer_pooling=False,
            cnn_conv_channels: list = None,
            classifier_hidden: int = 128,
            cnn_mixup_alpha: float = 0.2,
            cnn_use_mixup: bool = True,
            cnn_spec_augment: bool = True,
            cnn_freq_mask_param: int = 15,
            cnn_time_mask_param: int = 20,
            cnn_n_freq_masks: int = 2,
            cnn_n_time_masks: int = 2,
    ):
        super(CNNModel, self).__init__()

        self.conv_channels = cnn_conv_channels

        if self.conv_channels is None:
            self.conv_channels = [64, 128, 256]

        self.n_mels = n_mels
        self.last_layer_pooling = last_layer_pooling
        self.pooling = pooling
        self.spec_augment = cnn_spec_augment
        self.use_mixup = cnn_use_mixup
        self.mixup_alpha = cnn_mixup_alpha
        self.freq_mask_param = cnn_freq_mask_param
        self.time_mask_param = cnn_time_mask_param
        self.n_freq_masks = cnn_n_freq_masks
        self.n_time_masks = cnn_n_time_masks

        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
        )

        if self.spec_augment:
            from torchaudio.transforms import FrequencyMasking, TimeMasking
            self.freq_masks = nn.ModuleList([
                FrequencyMasking(self.freq_mask_param) for _ in range(self.n_freq_masks)
            ])
            self.time_masks = nn.ModuleList([
                TimeMasking(self.time_mask_param) for _ in range(self.n_time_masks)
            ])

        # Build CNN blocks dynamically
        blocks = []
        in_channels = 1
        for i, out_channels in enumerate(self.conv_channels):
            blocks.extend([
                nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(0.33),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.33),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            ])
            if i < len(self.conv_channels) - 1 or self.last_layer_pooling:
                blocks.append(nn.MaxPool2d(self.pooling[0], self.pooling[1]))
            blocks.append(nn.Dropout2d(dropout))
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.conv_channels[-1], classifier_hidden),
            nn.LeakyReLU(0.33),
            nn.Dropout(dropout*2),
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
            precomputed_mel: torch.Tensor = None,  # (BATCH, 1, n_mels, FRAMES)
    ) -> dict:
        # (BATCH, 1, n_mels, FRAMES)
        if precomputed_mel is not None:
            features = precomputed_mel
        else:
            with torch.no_grad():
                features = torch.log(self.mel_transform(audio_data) + 1e-6)

        if self.spec_augment and self.training:
            for freq_mask in self.freq_masks:
                features = freq_mask(features)
            for time_mask in self.time_masks:
                features = time_mask(features)

        # mixup during training
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


class DualChannelCNNModel(torch.nn.Module):
    """2-conv model from paper: processes 2 mel channels separately then concatenates."""

    def __init__(self,
                 sample_rate: int = 44100,
                 n_fft: int = 2048,
                 n_mels: int = 40,
                 dropout: float = 0.3,
                 n_label: int = 15,
                 pooling=[3, 3],
                 last_layer_pooling=False,
                 dccnn_conv_channels: list = None,
                 classifier_hidden: int = 1024,
                 dccnn_spec_augment=False,
                 dccnn_freq_mask_param=15,
                 dccnn_time_mask_param=20,
                 dccnn_n_freq_masks=2,
                 dccnn_n_time_masks=2,
                 dccnn_use_mixup=False,
                 dccnn_mixup_alpha=0.2,
                 ):
        super().__init__()

        self.conv_channels = dccnn_conv_channels

        if self.conv_channels is None:
            self.conv_channels = [32, 64, 128, 256]

        self.pooling = pooling
        self.last_layer_pooling = last_layer_pooling
        self.spec_augment = dccnn_spec_augment
        self.use_mixup = dccnn_use_mixup
        self.mixup_alpha = dccnn_mixup_alpha
        self.freq_mask_param = dccnn_freq_mask_param
        self.time_mask_param = dccnn_time_mask_param
        self.n_freq_masks = dccnn_n_freq_masks
        self.n_time_masks = dccnn_n_time_masks

        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
        )

        if self.spec_augment:
            from torchaudio.transforms import FrequencyMasking, TimeMasking
            self.freq_masks = nn.ModuleList([
                FrequencyMasking(self.freq_mask_param) for _ in range(self.n_freq_masks)
            ])
            self.time_masks = nn.ModuleList([
                TimeMasking(self.time_mask_param) for _ in range(self.n_time_masks)
            ])

        # two separate conv branches (one per channel)
        self.branch1 = self._make_branch(self.conv_channels, dropout)
        self.branch2 = self._make_branch(self.conv_channels, dropout)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # concat both branches -> 2x final conv channels
        self.classifier = nn.Sequential(
            nn.Linear(self.conv_channels[-1] * 2, classifier_hidden),
            nn.LeakyReLU(0.33),
            nn.Dropout(dropout*2),
            nn.Linear(classifier_hidden, n_label),
        )

    def _make_branch(self, conv_channels, dropout):
        blocks = []
        in_ch = 1
        for i, out_ch in enumerate(conv_channels):
            blocks.extend([
                nn.BatchNorm2d(in_ch),
                nn.LeakyReLU(0.33),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            ])

            if i < len(self.conv_channels) - 1 or self.last_layer_pooling:
                blocks.append(nn.MaxPool2d(self.pooling[0], self.pooling[1]))

            blocks.append(nn.Dropout2d(dropout))
            in_ch = out_ch
        return nn.Sequential(*blocks)

    def mixup_dual(self, x1, x2, y):
        lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
        idx = torch.randperm(x1.size(0), device=x1.device)

        x1_mix = lam * x1 + (1 - lam) * x1[idx]
        x2_mix = lam * x2 + (1 - lam) * x2[idx]

        return x1_mix, x2_mix, y, y[idx], lam

    def forward(
            self,
            audio_ch1: torch.Tensor,
            audio_ch2: torch.Tensor,
            labels=None,
            precomputed_mel1: torch.Tensor = None,
            precomputed_mel2: torch.Tensor = None,
    ):
        # each input: (B, 1, TIME)
        if precomputed_mel1 is not None and precomputed_mel2 is not None:
            mel1 = precomputed_mel1
            mel2 = precomputed_mel2
        else:
            with torch.no_grad():
                mel1 = torch.log(self.mel_transform(audio_ch1) + 1e-6)  # (B, 1, mels, frames)
                mel2 = torch.log(self.mel_transform(audio_ch2) + 1e-6)

        if self.spec_augment and self.training:
            for freq_mask in self.freq_masks:
                mel1 = freq_mask(mel1)
                mel2 = freq_mask(mel2)
            for time_mask in self.time_masks:
                mel1 = time_mask(mel1)
                mel2 = time_mask(mel2)

        # mixup during training
        if self.use_mixup and self.training and labels is not None:
            mel1, mel2, y_a, y_b, lam = self.mixup_dual(mel1, mel2, labels)

        feat1 = self.global_pool(self.branch1(mel1)).flatten(1)
        feat2 = self.global_pool(self.branch2(mel2)).flatten(1)

        combined = torch.cat([feat1, feat2], dim=1)
        logits = self.classifier(combined)

        return {"logits": logits}


class EnsembleCNNModel(torch.nn.Module):
    def __init__(self, cnn_config: dict, dccnn_config: dict, sample_rate: int = 44100, shared_mel: bool = True):
        super().__init__()

        self.preprocessor = MultiStreamPreprocessor(sample_rate=sample_rate)
        self.shared_mel = shared_mel
        if self.shared_mel:
            if cnn_config.get('n_mels') != dccnn_config.get('n_mels'):
                raise ValueError("shared_mel requires matching n_mels for cnn and dccnn configs")
            self.mel_transform = MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=cnn_config.get('n_fft', 2048),
                f_min=cnn_config.get('f_min', 0.0),
                f_max=cnn_config.get('f_max', 22050.0),
                n_mels=cnn_config.get('n_mels', 40),
            )

        self.models = nn.ModuleDict({
            'stereo': DualChannelCNNModel(**dccnn_config),
            # 'right': DualChannelCNNModel(**dccnn_config),
            'ms': DualChannelCNNModel(**dccnn_config),
            # 'side': DualChannelCNNModel(**dccnn_config),
            'harmonic': CNNModel(**cnn_config),
            'percussive': CNNModel(**cnn_config),
            'background': CNNModel(**cnn_config),
            # 'foreground': CNNModel(**cnn_config)
        })

        self.ensemble_weights = nn.Parameter(torch.ones(len(self.models)))

    def forward(self, audio_stereo: torch.Tensor, labels=None):
        # audio_stereo: (B, 2, TIME) or dict from dataset

        batch_streams = None
        log_mels = {}
        if isinstance(audio_stereo, dict):
            if 'mels' in audio_stereo:
                log_mels = audio_stereo['mels']
            elif 'streams' in audio_stereo:
                batch_streams = audio_stereo['streams']
            else:
                raise RuntimeError("multi_stream forward expects 'streams' or 'mels'")
        else:
            raise RuntimeError("multi_stream forward expects a dict batch")

        all_logits = []
        if self.shared_mel and batch_streams is not None:
            for key, stream in batch_streams.items():
                with torch.no_grad():
                    log_mels[key] = torch.log(self.mel_transform(stream) + 1e-6)

        # 1. Dual Channel models (require 2 inputs)
        pairs = {
            'stereo': ('left', 'right'),
            'ms': ('mid', 'side'),
        }

        for name, (s1_key, s2_key) in pairs.items():
            if log_mels:
                out = self.models[name](None, None, labels, log_mels[s1_key], log_mels[s2_key])
            else:
                s1 = batch_streams[s1_key]
                s2 = batch_streams[s2_key]
                out = self.models[name](s1, s2, labels)
            all_logits.append(out['logits'])

        # 2. Single Channel models
        for name in ['harmonic', 'percussive', 'background']:
            if log_mels:
                out = self.models[name](None, labels, precomputed_mel=log_mels[name])
            else:
                out = self.models[name](batch_streams[name], labels)
            all_logits.append(out['logits'])

        logits_stack = torch.stack(all_logits)  # (num_models, B, num_classes)
        weights = torch.softmax(self.ensemble_weights, dim=0)  # (num_models,)
        weighted_logits = logits_stack * weights[:, None, None]
        return {'logits': weighted_logits.sum(dim=0)} # WEIGH MODELS


class CNNTCNModel(nn.Module):
    """
    CNN + TCN model for acoustic scene classification.
    - Extracts log-mel spectrograms
    - 2D CNN for local time-freq patterns
    - TCN for temporal context
    - SpecAugment and mixup supported
    """

    def __init__(
            self,
            sample_rate: int = 44100,
            n_fft: int = 2048,
            n_mels: int = 128,
            f_min: float = 0.0,
            f_max: float = 22050.0,

            cnn_channels: list = None,
            cnn_kernel_size: tuple = (3, 3),
            cnn_pool_size: tuple = (2, 2),

            tcn_channels: list = None,
            tcn_kernel_size: int = 3,

            dropout: float = 0.3,

            classifier_hidden: int = 128,
            n_label: int = 15,

            spec_augment: bool = True,
            freq_mask_param: int = 20,
            time_mask_param: int = 30,
            n_freq_masks: int = 2,
            n_time_masks: int = 2,
            use_mixup: bool = True,
            mixup_alpha: float = 0.3,
    ):
        super(CNNTCNModel, self).__init__()

        if cnn_channels is None:
            cnn_channels = [32, 64]
        if tcn_channels is None:
            tcn_channels = [64, 64]

        self.spec_augment = spec_augment
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
        )

        # SpecAugment
        if self.spec_augment:
            from torchaudio.transforms import FrequencyMasking, TimeMasking
            self.freq_masks = nn.ModuleList([
                FrequencyMasking(freq_mask_param)
                for _ in range(n_freq_masks)
            ])
            self.time_masks = nn.ModuleList([
                TimeMasking(time_mask_param)
                for _ in range(n_time_masks)
            ])

        # 2D CNN feature extraction blocks
        self.cnn_blocks = nn.ModuleList()
        in_channels = 1
        for out_channels in cnn_channels:
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=cnn_kernel_size, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=cnn_kernel_size, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(cnn_pool_size),
                nn.Dropout2d(dropout),
            )
            self.cnn_blocks.append(block)
            in_channels = out_channels

        self.tcn_blocks = nn.ModuleList()

        # Pool freq-axis
        self.adaptive_freq_pool = nn.AdaptiveAvgPool2d((8, None))  # Reduce freq to 8 bins
        tcn_input_channels = cnn_channels[-1] * 8  # cnn_channels[-1] * 8 (freq bins after adaptive pool)

        in_channels = tcn_input_channels
        for i, out_channels in enumerate(tcn_channels):
            dilation = 2 ** i
            block = TCNBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=tcn_kernel_size,
                dilation=dilation,
                dropout=dropout,
            )
            self.tcn_blocks.append(block)
            in_channels = out_channels

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(tcn_channels[-1], classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, n_label),
        )

    def mixup_data(self, x, y):
        lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def forward(self, audio_data: torch.Tensor, labels: torch.Tensor = None) -> dict:
        with torch.no_grad():
            features = torch.log(self.mel_transform(audio_data) + 1e-6)

        # SpecAugment
        if self.spec_augment and self.training:
            for freq_mask in self.freq_masks:
                features = freq_mask(features)
            for time_mask in self.time_masks:
                features = time_mask(features)

        # mixup during training
        y_a, y_b, lam = None, None, None
        if self.use_mixup and self.training and labels is not None:
            features, y_a, y_b, lam = self.mixup_data(features, labels)

        # CNN Blocks
        x = features
        for cnn_block in self.cnn_blocks:
            x = cnn_block(x)

        # Adaptive pooling and reshape for TCN
        x = self.adaptive_freq_pool(x)  # (B, C, F=8, T)
        batch, channels, freq, time = x.shape
        x = x.view(batch, channels * freq, time)

        # TCN Blocks
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)

            # Global pooling over time
        x = self.global_pool(x).squeeze(-1)

        logits = self.classifier(x)

        if self.training and y_a is not None:
            return {"logits": logits, "y_a": y_a, "y_b": y_b, "lam": lam}
        return {"logits": logits}


class TCNBlock(nn.Module):
    """
    1D Temporal Convolutional Network block with residual connection.
    - Two (dilated conv + BN + ReLU + dropout) stages
    - Residual path with 1x1 conv if needed
    - Non-causal (pads both sides)
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            dilation: int = 1,
            dropout: float = 0.2,
    ):
        super(TCNBlock, self).__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Project input if channel dims differ
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        return self.relu(out + residual)


class SklearnAudioClassifier:
    """Wrapper for sklearn classifiers on mel spectrograms."""

    def __init__(
            self,
            classifier_type: str = "random_forest",
            sample_rate: int = 44100,
            n_fft: int = 2048,
            hop_length: int = 882,
            n_mels: int = 40,
            n_estimators: int = 500,
            max_depth: int = 20,
            n_components: int = 100,
            svm_C: float = 10.0,
            svm_kernel: str = "rbf",
            win_length: int = 2048,
    ):
        self.win_length = win_length
        self.classifier_type = classifier_type
        self.n_mels = n_mels

        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        if classifier_type == "random_forest":
            self.pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    n_jobs=-1,
                    random_state=42
                ))
            ])
        elif classifier_type == "pca_svm":
            self.pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components)),
                ("clf", SVC(kernel=svm_kernel, C=svm_C))
            ])
        else:
            raise ValueError(f"Unknown classifier: {classifier_type}")

    def extract_features(self, audio_data: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            window = torch.hann_window(self.win_length)
            mel_spec = torch.log(self.mel_transform(audio_data) + 1e-6)
            mel_spec = mel_spec.squeeze(1).numpy()

            centroid = torchaudio.functional.spectral_centroid(
                audio_data, 44100, pad=0, window=window, n_fft=2048, hop_length=882, win_length=2048
            ).squeeze(1)

        features = []
        for i, spec in enumerate(mel_spec):
            feat = np.concatenate([
                spec.mean(axis=1),
                np.median(spec, axis=1),
                spec.std(axis=1),
                spec.max(axis=1),
                spec.min(axis=1),
                np.ptp(spec, axis=1),
                skew(spec, axis=1),
                scipy.stats.kurtosis(spec, axis=1),
                np.array([centroid[i].mean()], dtype=np.float32),
                np.array([centroid[i].std()], dtype=np.float32),
                np.sqrt(np.mean(spec ** 2, axis=1)),
            ])
            features.append(feat)
        return np.array(features)

    def fit(self, dataloader):
        X_all, y_all = [], []
        for batch in tqdm(dataloader, desc="Fitting classifier"):
            audio = batch['audio_data']
            labels = batch['class_label'].numpy()
            X_all.append(self.extract_features(audio))
            y_all.append(labels)

        X = np.vstack(X_all)
        y = np.concatenate(y_all)
        self.pipeline.fit(X, y)
        return self

    def predict(self, dataloader):
        X_all, y_all = [], []
        for batch in tqdm(dataloader, desc="Predicting"):
            audio = batch['audio_data']
            labels = batch['class_label'].numpy()
            X_all.append(self.extract_features(audio))
            y_all.append(labels)
        X = np.vstack(X_all)
        y_true = np.concatenate(y_all)
        y_pred = self.pipeline.predict(X)
        return y_pred, y_true

    def score(self, dataloader):
        y_pred, y_true = self.predict(dataloader)
        return (y_pred == y_true).mean()


class SklearnAudioEnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            base_classifiers=None,
            final_estimator=None,
            n_mels=40,
            sample_rate=44100,
            n_fft=2048,
            hop_length=882,
            win_length: int = 2048,
            ensemble_type="stacking",
    ):
        self.base_classifiers = base_classifiers
        self.final_estimator = final_estimator
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mel_transform = None
        self.ensemble_type = ensemble_type
        self.mel_transform = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        self.win_length = win_length

    def fit(self, dataloader):
        X_all, y_all = [], []
        for batch in tqdm(dataloader, desc="Fitting ensemble classifier"):
            audio = batch['audio_data']
            labels = batch['class_label'].numpy()
            X_all.append(self.extract_features(audio))
            y_all.append(labels)
        X = np.vstack(X_all)
        y = np.concatenate(y_all)

        estimators = []
        for clf_cfg in self.base_classifiers:
            name = clf_cfg['name']
            params = clf_cfg['params']
            if name == "random_forest":
                clf = RandomForestClassifier(**params)
            elif name == "pca_svm":
                clf = Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=params['n_components'])),
                    ("clf", SVC(kernel=params['svm_kernel'], C=params['svm_C']))
                ])
            else:
                raise ValueError(f"Unknown classifier: {name}")
            estimators.append((name, clf))

        if self.ensemble_type == "stacking":
            self.ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(**self.final_estimator['params'])
            )
        elif self.ensemble_type == "voting":
            self.ensemble = VotingClassifier(
                estimators=estimators,
                voting='hard'
            )
        else:
            raise ValueError(f"Unknown ensemble_type: {self.ensemble_type}")

        self.ensemble.fit(X, y)
        return self

    def extract_features(self, audio_data: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            window = torch.hann_window(self.win_length)
            mel_spec = torch.log(self.mel_transform(audio_data) + 1e-6)
            mel_spec = mel_spec.squeeze(1).numpy()

            centroid = torchaudio.functional.spectral_centroid(
                audio_data, 44100, pad=0, window=window, n_fft=2048, hop_length=882, win_length=2048
            ).squeeze(1)

        features = []
        for i, spec in enumerate(mel_spec):
            feat = np.concatenate([
                spec.mean(axis=1),
                np.median(spec, axis=1),
                spec.std(axis=1),
                spec.max(axis=1),
                spec.min(axis=1),
                np.ptp(spec, axis=1),
                skew(spec, axis=1),
                scipy.stats.kurtosis(spec, axis=1),
                np.array([centroid[i].mean()], dtype=np.float32),
                np.array([centroid[i].std()], dtype=np.float32),
                np.sqrt(np.mean(spec ** 2, axis=1)),
            ])
            features.append(feat)
        return np.array(features)

    def predict(self, dataloader_or_X):
        """
        Supports two calling conventions:
            1. predict(dataloader) -> (y_pred, y_true)
            2. predict(X) where X is a feature matrix -> y_pred
        """
        if isinstance(dataloader_or_X, np.ndarray):
            X = dataloader_or_X
            return self.ensemble.predict(X)

        if torch.is_tensor(dataloader_or_X):
            X = dataloader_or_X.detach().cpu().numpy()
            return self.ensemble.predict(X)

        dataloader = dataloader_or_X
        X_all, y_all = [], []
        for batch in tqdm(dataloader, desc="Predicting"):
            audio = batch['audio_data']
            labels = batch['class_label'].numpy()
            X_all.append(self.extract_features(audio))
            y_all.append(labels)

        X = np.vstack(X_all)
        y_true = np.concatenate(y_all)
        y_pred = self.ensemble.predict(X)
        return y_pred, y_true

    def score(self, dataloader):
        y_pred, y_true = self.predict(dataloader)
        return (y_pred == y_true).mean()
