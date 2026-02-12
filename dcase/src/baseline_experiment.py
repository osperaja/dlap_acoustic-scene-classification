import torch

try:
    from .experiment import AcousticScenesExperiment
except ImportError:
    from experiment import AcousticScenesExperiment
from torchmetrics import Accuracy
from torchmetrics.aggregation import RunningMean
from typing import Literal


class BaselineExperiment(AcousticScenesExperiment):
    def __init__(
            self,
            model: torch.nn.Module,
            n_label: int,
            **exp_kwargs,
    ):
        super(BaselineExperiment, self).__init__(**exp_kwargs)

        # init attributes
        self.model = model

        # init metrics
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.accuracy = Accuracy(task='multiclass', num_classes=n_label)
        self.running_accuracy = RunningMean(window=100)

    def shared_step(self, batch, batch_idx, stage: Literal['train', 'val', 'test']):
        # load data
        audio_data = batch['audio_data'].to(device=self.device)  # (BATCH, CHANNEL, TIME)
        target_label = batch['class_label'].to(device=self.device)  # (BATCH)

        # forward model
        logits = self.model(audio_data)  # (BATCH, FRAMES', CLASS)

        # compute loss
        t_label = target_label[:, None].expand(logits.shape[:-1])  # (BATCH, FRAMES')
        loss = self.ce_loss(
            torch.flatten(logits, end_dim=1), torch.flatten(t_label, end_dim=1)
        )

        # compute accuracy with majority voting
        with torch.no_grad():
            est_label = torch.argmax(logits, dim=-1).squeeze(dim=1)  # (BATCH, FRAMES')
            est_label = torch.stack(
                [
                    torch.bincount(e_label).argmax() for e_label in est_label
                ]
            )  # (BATCH)
            est_accuracy = self.accuracy(est_label, target_label)

            # loss and accuracy logging
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
