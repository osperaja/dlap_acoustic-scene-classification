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

        # Forward model - pass labels for mixup during training
        labels_for_mixup = target_label if stage == 'train' else None
        
        if 'streams' in batch:
            # Pass the streams dict directly if it exists (from multi_stream dataset)
            output = self.model(batch, labels_for_mixup)
        else:
            audio_data = batch['audio_data']  # (BATCH, CHANNEL, TIME)
            output = self.model(audio_data, labels_for_mixup)

        # Handle dict output from CNNModel
        logits = output["logits"]  # (BATCH, n_label)

        # Compute loss
        if stage == 'train' and "y_a" in output:
            loss = self.mixup_criterion(logits, output["y_a"], output["y_b"], output["lam"])
        else:
            loss = self.ce_loss(logits, target_label)

        # Compute accuracy
        with torch.no_grad():
            est_label = torch.argmax(logits, dim=-1)  # (BATCH)
            est_accuracy = self.accuracy(est_label, target_label)

        self.log(
            f'{stage}/loss',
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            prog_bar=False
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
                prog_bar=False
            )
            self.log(
                f'{stage}/running_accuracy',
                self.running_accuracy.compute(),
                on_step=True,
                on_epoch=False,
                logger=True,
                sync_dist=True,
                prog_bar=True
            )
        elif stage == 'val':
            self.log(
                f'{stage}/accuracy',
                est_accuracy,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                prog_bar=False
            )
        return loss
