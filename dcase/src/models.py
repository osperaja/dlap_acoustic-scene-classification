import os

import numpy as np
import scipy
import torch
import torchaudio
from scipy.stats import skew
from sklearn.linear_model import LogisticRegression
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

import random
from typing import Optional

def _set_and_print_seed(owner_name: str, seed: Optional[int] = None) -> int:
    """Set torch / numpy / python random seeds and print the chosen seed."""
    if seed is None:
        seed = np.random.randint(0, 2**31 - 1)
    seed = int(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    np.random.seed(seed)
    random.seed(seed)
    print(f"seed for {owner_name}: {seed}")
    return seed


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
            random_seed: Optional[int] = None,
    ):
        super(BaselineModel, self).__init__()
        self.seed = _set_and_print_seed(self.__class__.__name__, random_seed)

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
            spec_augment: bool = False,
            freq_mask_param: int = 10,
            time_mask_param: int = 20,
            n_freq_masks: int = 2,
            n_time_masks: int = 2,
            random_seed: Optional[int] = None,
    ):
        super(LinSeqModel, self).__init__()
        self.seed = _set_and_print_seed(self.__class__.__name__, random_seed)

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

        layers = [nn.Linear(n_mels, n_hidden_feats), nn.ReLU(), nn.Dropout(dropout)]

        for _ in range(n_hidden_layer - 1):
            layers.append(nn.Linear(n_hidden_feats, n_hidden_feats))
            layers.append(nn.ReLU())
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
    CNN for spectrogram classification with SpecAugment.
    """

    def __init__(
            self,
            n_mels: int,
            dropout: float,
            n_label: int,
            pooling,
            last_layer_pooling,
            conv_channels: list,
            classifier_hidden: int,
            spec_augment: bool,
            freq_mask_param: int,
            time_mask_param: int,
            n_freq_masks: int,
            n_time_masks: int,
            use_mixup: bool,
            mixup_alpha: float,
            use_classifier: bool,
            random_seed: Optional[int] = None,
            sample_rate: int = 44100,
            n_fft: int = 2048,
            f_min: float = 0.0,
            f_max: float = 22050.0,
    ):
        super(CNNModel, self).__init__()
        self.seed = _set_and_print_seed(self.__class__.__name__, random_seed)
        self.use_classifier = use_classifier

        self.conv_channels = conv_channels

        if self.conv_channels is None:
            self.conv_channels = [64, 128, 256]

        self.n_mels = n_mels
        self.last_layer_pooling = last_layer_pooling
        self.pooling = pooling
        self.spec_augment = spec_augment
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

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
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        if use_classifier: self.classifier = nn.Sequential(
            nn.Linear(self.conv_channels[-1], classifier_hidden),
            nn.LeakyReLU(0.33),
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
            precomputed_mel: torch.Tensor = None,  # (BATCH, 1, n_mels, FRAMES)
    ) -> dict:
        if precomputed_mel is not None:
            features = precomputed_mel
        else:
            with torch.no_grad():
                features = torch.log(self.mel_transform(audio_data.float()) + 1e-6)

        if self.spec_augment and self.training:
            for freq_mask in self.freq_masks:
                features = freq_mask(features)
            for time_mask in self.time_masks:
                features = time_mask(features)

        y_a, y_b, lam = None, None, None
        if self.use_mixup and self.training and labels is not None:
            features, y_a, y_b, lam = self.mixup_data(features, labels)

        x = self.conv_blocks(features)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        if not self.use_classifier:
            return {"features": x}

        logits = self.classifier(x)

        if self.training and y_a is not None:
            return {"logits": logits, "y_a": y_a, "y_b": y_b, "lam": lam}
        return {"logits": logits}


class DualChannelCNNModel(torch.nn.Module):
    """2-conv model from paper: processes 2 mel channels separately then concatenates."""

    def __init__(self,
                 n_mels: int,
                 dropout: float,
                 n_label: int,
                 pooling,
                 last_layer_pooling,
                 conv_channels: list,
                 classifier_hidden: int,
                 spec_augment,
                 freq_mask_param,
                 time_mask_param,
                 n_freq_masks,
                 n_time_masks,
                 use_mixup,
                 mixup_alpha,
                 use_classifier: bool,
                 random_seed: Optional[int] = None,
                 sample_rate: int = 44100,
                 n_fft: int = 2048,
                 ):
        super().__init__()

        self.seed = _set_and_print_seed(self.__class__.__name__, random_seed)

        self.conv_channels = conv_channels
        self.use_classifier = use_classifier

        if self.conv_channels is None:
            self.conv_channels = [32, 64, 128, 256]

        self.preprocessor = MultiStreamPreprocessor(sample_rate=sample_rate)

        self.pooling = pooling
        self.last_layer_pooling = last_layer_pooling
        self.spec_augment = spec_augment
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

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
        self.branch1 = self._make_branch(self.conv_channels)
        self.branch2 = self._make_branch(self.conv_channels)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # concat both branches -> 2x final conv channels
        if use_classifier: self.classifier = nn.Sequential(
            nn.Linear(self.conv_channels[-1] * 2, classifier_hidden),
            nn.LeakyReLU(0.33),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, n_label),
        )
        else: self.classifier = None

    def _make_branch(self, conv_channels):
        blocks = []
        in_ch = 1
        for i, out_ch in enumerate(conv_channels):
            blocks.extend([
                nn.BatchNorm2d(in_ch),
                nn.LeakyReLU(0.33),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.33),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            ])

            if i < len(conv_channels) - 1 or self.last_layer_pooling:
                blocks.append(nn.MaxPool2d(self.pooling[0], self.pooling[1]))
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
        if precomputed_mel1 is not None and precomputed_mel2 is not None:
            mel1 = precomputed_mel1
            mel2 = precomputed_mel2
        else:
            with torch.no_grad():
                mel1 = torch.log(self.mel_transform(audio_ch1.float()) + 1e-6)
                mel2 = torch.log(self.mel_transform(audio_ch2.float()) + 1e-6)

        # DEBUG: Print shapes and stats once
        if not hasattr(self, '_debug_done'):
            print(f"\n[DEBUG DualChannelCNN]")
            print(f"  audio_ch1 input: {audio_ch1.shape}, min={audio_ch1.min():.4f}, max={audio_ch1.max():.4f}")
            print(f"  audio_ch2 input: {audio_ch2.shape}, min={audio_ch2.min():.4f}, max={audio_ch2.max():.4f}")
            print(f"  mel1 shape: {mel1.shape}, min={mel1.min():.2f}, max={mel1.max():.2f}, mean={mel1.mean():.2f}")
            print(f"  mel2 shape: {mel2.shape}, min={mel2.min():.2f}, max={mel2.max():.2f}, mean={mel2.mean():.2f}")
            self._debug_done = True

        if self.spec_augment and self.training:
            for freq_mask in self.freq_masks:
                mel1 = freq_mask(mel1)
                mel2 = freq_mask(mel2)
            for time_mask in self.time_masks:
                mel1 = time_mask(mel1)
                mel2 = time_mask(mel2)

        # Initialize mixup variables
        y_a, y_b, lam = None, None, None

        if self.use_mixup and self.training and labels is not None:
            mel1, mel2, y_a, y_b, lam = self.mixup_dual(mel1, mel2, labels)

        feat1 = self.global_pool(self.branch1(mel1)).flatten(1)
        feat2 = self.global_pool(self.branch2(mel2)).flatten(1)
        combined = torch.cat([feat1, feat2], dim=1)

        # DEBUG: Check feature stats
        if not hasattr(self, '_debug_feat_done'):
            print(f"  feat1: {feat1.shape}, min={feat1.min():.4f}, max={feat1.max():.4f}, mean={feat1.mean():.4f}")
            print(f"  feat2: {feat2.shape}, min={feat2.min():.4f}, max={feat2.max():.4f}, mean={feat2.mean():.4f}")
            self._debug_feat_done = True

        if not self.use_classifier:
            return {"features": combined}

        logits = self.classifier(combined)

        # DEBUG: Check logits
        if not hasattr(self, '_debug_logits_done'):
            print(f"  logits: {logits.shape}, min={logits.min():.4f}, max={logits.max():.4f}")
            print(f"  logits sample[0]: {logits[0].detach().cpu().numpy()}")
            self._debug_logits_done = True

        # Return mixup info when mixup was applied
        if self.training and y_a is not None:
            return {"logits": logits, "y_a": y_a, "y_b": y_b, "lam": lam}
        return {"logits": logits}


class EnsembleCNNModel(torch.nn.Module):
    def __init__(
            self,
            cnn_config: dict,
            dccnn_config: dict,
            sample_rate: int,
            shared_mel: bool,
            classifier_hidden: int,
            dropout: float,
            random_seed: Optional[int] = None,
            pretrained_checkpoints: dict = None,
            freeze_submodels: bool = True,
    ):
        super().__init__()
        base_seed = _set_and_print_seed(self.__class__.__name__, random_seed)

        self.preprocessor = MultiStreamPreprocessor(sample_rate=sample_rate)
        self.shared_mel = shared_mel
        self.freeze_submodels = freeze_submodels

        if self.shared_mel:
            if cnn_config.get('n_mels') != dccnn_config.get('n_mels'):
                raise ValueError("shared_mel requires matching n_mels")
            self.mel_transform = MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=cnn_config.get('n_fft', 2048),
                f_min=cnn_config.get('f_min', 0.0),
                f_max=cnn_config.get('f_max', 22050.0),
                n_mels=cnn_config.get('n_mels', 128),
            )

        # Force use_classifier=False for ensemble sub-models (we use shared classifier)
        # cnn_config_ensemble = {**cnn_config, 'use_classifier': False}
        dccnn_config_ensemble = {**dccnn_config, 'use_classifier': False}

        self.models = nn.ModuleDict({
            'stereo': DualChannelCNNModel(**{**dccnn_config_ensemble, 'random_seed': base_seed}),
            'ms': DualChannelCNNModel(**{**dccnn_config_ensemble, 'random_seed': base_seed + 1}),
            'hpss': DualChannelCNNModel(**{**dccnn_config_ensemble, 'random_seed': base_seed + 2}),
        })

        # Load pretrained checkpoints if provided
        if pretrained_checkpoints:
            self._load_pretrained_checkpoints(pretrained_checkpoints)

        # Freeze sub-models if requested (train only classifier)
        if self.freeze_submodels and pretrained_checkpoints:
            self._freeze_submodels()

        # cnn_feat_dim = cnn_config.get('conv_channels')[-1]
        dccnn_feat_dim = dccnn_config.get('conv_channels')[-1] * 2
        n_label = cnn_config.get('n_label', 15)

        total_feat_dim = (dccnn_feat_dim * 3)

        self.classifier = nn.Sequential(
            nn.Linear(total_feat_dim, classifier_hidden),
            nn.LeakyReLU(0.33),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, n_label),
        )

    def _load_pretrained_checkpoints(self, checkpoints: dict):
        """
        Load pretrained weights for sub-models.
        """
        for name, ckpt_path in checkpoints.items():
            if name not in self.models:
                print(f"Warning: Unknown model name '{name}', skipping")
                continue

            if ckpt_path is None:
                print(f"No checkpoint for '{name}', using random init")
                continue

            print(f"Loading pretrained checkpoint for '{name}' from {ckpt_path}")

            # Load the full experiment checkpoint
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            state_dict = checkpoint.get('state_dict', checkpoint)

            # Extract only the model weights (remove 'model.' prefix from experiment)
            model_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    new_key = key[6:]  # Remove 'model.' prefix
                    # Skip classifier weights (we use our own)
                    if not new_key.startswith('classifier'):
                        model_state_dict[new_key] = value

            # Load weights into sub-model
            missing, unexpected = self.models[name].load_state_dict(
                model_state_dict, strict=False
            )
            if missing:
                print(f"  Missing keys: {missing}")
            if unexpected:
                print(f"  Unexpected keys: {unexpected}")

    def _freeze_submodels(self):
        """Freeze all sub-model parameters, only classifier will be trained."""
        for name, model in self.models.items():
            for param in model.parameters():
                param.requires_grad = False
            print(f"Froze parameters for '{name}'")

    def unfreeze_submodels(self):
        """Unfreeze sub-models for fine-tuning."""
        for name, model in self.models.items():
            for param in model.parameters():
                param.requires_grad = True
            print(f"Unfroze parameters for '{name}'")

    def forward(self, audio_ch1, audio_ch2, audio_ch3, audio_ch4, audio_ch5, audio_ch6, labels=None):
        all_logits = []

        channel_pairs = {
            'stereo': (audio_ch1, audio_ch2),
            'ms': (audio_ch3, audio_ch4),
            'hpss': (audio_ch5, audio_ch6),
        }

        for name, (ch1, ch2) in channel_pairs.items():
            out = self.models[name](
                audio_ch1=ch1,
                audio_ch2=ch2,
                labels=labels,
            )
            all_logits.append(out['features'])

        combined = torch.cat(all_logits, dim=1)
        logits = self.classifier(combined)

        return {'logits': logits}

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

            sample_rate: int,
            n_fft: int,
            n_mels: int,
            f_min: float,
            f_max: float,

            cnn_channels: list,
            cnn_kernel_size: tuple,
            cnn_pool_size: tuple,

            tcn_channels: list,
            tcn_kernel_size: int,

            dropout: float,

            classifier_hidden: int,
            n_label: int,

            spec_augment: bool,
            freq_mask_param: int,
            time_mask_param: int,
            n_freq_masks: int,
            n_time_masks: int,
            use_mixup: bool,
            mixup_alpha: float,
            random_seed: Optional[int] = None,
    ):
        super(CNNTCNModel, self).__init__()
        self.seed = _set_and_print_seed(self.__class__.__name__, random_seed)

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
                nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(0.33),
                nn.Conv2d(in_channels, out_channels, kernel_size=cnn_kernel_size, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.33),
                nn.Conv2d(out_channels, out_channels, kernel_size=cnn_kernel_size, padding=1),
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

        self.lrelu = nn.LeakyReLU(0.33)
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
        out = self.lrelu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.dropout(out)

        return self.lrelu(out + residual)


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
