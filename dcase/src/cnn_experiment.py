import time
import torch

try:
    from .experiment import AcousticScenesExperiment
except ImportError:
    from experiment import AcousticScenesExperiment
from torchmetrics import Accuracy
from torchmetrics.aggregation import RunningMean
from typing import Literal


class CNNExperiment(AcousticScenesExperiment):
    def __init__(
            self,
            model: torch.nn.Module,
            n_label: int,
            **exp_kwargs,
    ):
        super(CNNExperiment, self).__init__(**exp_kwargs)

        self.model = model

        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.accuracy = Accuracy(task='multiclass', num_classes=n_label)
        self.running_accuracy = RunningMean(window=100)

    def _move_to_device(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(device=self.device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._move_to_device(v) for v in obj]
        return obj

    def mixup_criterion(self, logits, y_a, y_b, lam):
        """Compute mixup loss as weighted combination of two targets."""
        return lam * self.ce_loss(logits, y_a) + (1 - lam) * self.ce_loss(logits, y_b)

    def shared_step(self, batch, batch_idx, stage: Literal['train', 'val', 'test']):
        step_start = time.perf_counter()
        batch = self._move_to_device(batch)
        target_label = batch['class_label']  # (BATCH)

        # Get batch size for logging
        batch_size = target_label.size(0)

        # Forward model - pass labels for mixup during training
        labels_for_mixup = target_label if stage == 'train' else None

        # Determine input format and call appropriate forward
        if 'streams' in batch or 'mels' in batch:
            output = self.model(batch, labels_for_mixup)
        elif all(key in batch for key in ['audio_ch1', 'audio_ch2', 'audio_ch3', 'audio_ch4', 'audio_ch5', 'audio_ch6']):
            output = self.model(
                batch['audio_ch1'],
                batch['audio_ch2'],
                batch['audio_ch3'],
                batch['audio_ch4'],
                batch['audio_ch5'],
                batch['audio_ch6'],
                labels_for_mixup
            )
        elif 'audio_ch1' in batch and 'audio_ch2' in batch:
            output = self.model(
                batch['audio_ch1'],
                batch['audio_ch2'],
                labels_for_mixup
            )
        else:
            audio_data = batch['audio_data']
            output = self.model(audio_data, labels_for_mixup)

        logits = output["logits"]

        # Compute loss
        if stage == 'train' and "y_a" in output:
            loss = self.mixup_criterion(logits, output["y_a"], output["y_b"], output["lam"])
        else:
            loss = self.ce_loss(logits, target_label)

        # Compute accuracy
        with torch.no_grad():
            est_label = torch.argmax(logits, dim=-1)
            est_accuracy = self.accuracy(est_label, target_label)

        # Add batch_size to all log calls!
        self.log(
            f'{stage}/loss',
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            prog_bar=False,
            batch_size=batch_size
        )
        if stage == 'train':
            self.running_accuracy(est_accuracy)
            compute_time = time.perf_counter() - step_start
            self._last_compute_time = compute_time
            self.log(
                'train/compute_time',
                compute_time,
                on_step=True,
                on_epoch=False,
                logger=True,
                sync_dist=True,
                prog_bar=False,
                batch_size=batch_size
            )
            self.log(
                f'{stage}/running_accuracy',
                self.running_accuracy.compute(),
                on_step=True,
                on_epoch=False,
                logger=True,
                sync_dist=True,
                prog_bar=True,
                batch_size=batch_size
            )
        elif stage == 'val':
            self.log(
                f'{stage}/accuracy',
                est_accuracy,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                prog_bar=False,
                batch_size=batch_size
            )
        return loss

