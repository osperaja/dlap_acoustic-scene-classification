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

    def mixup_criterion(self, logits, y_a, y_b, lam):
        """Compute mixup loss as weighted combination of two targets."""
        return lam * self.ce_loss(logits, y_a) + (1 - lam) * self.ce_loss(logits, y_b)

    def shared_step(self, batch, batch_idx, stage: Literal['train', 'val', 'test']):
        audio_data = batch['audio_data'].to(device=self.device)  # (BATCH, CHANNEL, TIME)
        target_label = batch['class_label'].to(device=self.device)  # (BATCH)

        # Forward model - pass labels for mixup during training
        labels_for_mixup = target_label if stage == 'train' else None
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
