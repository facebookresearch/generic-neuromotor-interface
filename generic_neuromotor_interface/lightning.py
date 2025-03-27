# pyre-strict
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Mapping

import numpy as np

import pytorch_lightning as pl
import torch

from generic_neuromotor_interface.cler import compute_cler, GestureType
from generic_neuromotor_interface.handwriting_utils import (
    CharacterErrorRates,
    charset,
    Decoder,
)
from torch import nn
from torchmetrics import MetricCollection

log = logging.getLogger(__name__)


class BaseLightningModule(pl.LightningModule):
    """Child classes should implement _step."""

    def __init__(self, network: nn.Module, optimizer: torch.optim.Optimizer) -> None:

        super().__init__()
        self.save_hyperparameters()
        self.network = network
        self.optimizer = optimizer

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        return self.network(emg)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="val")

    def test_step(
        self, batch, batch_idx, dataloader_idx: int | None = None
    ) -> torch.Tensor:
        return self._step(batch, stage="test")

    def configure_optimizers(self):
        return self.optimizer(self.parameters())


class WristModule(BaseLightningModule):
    def __init__(self, network: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        super().__init__(network=network, optimizer=optimizer)
        self.loss_fn = torch.nn.L1Loss(reduction="mean")

    def _step(self, batch: Mapping[str, torch.Tensor], stage: str = "train") -> float:

        # Extract data
        emg = batch["emg"]
        wrist_angles = batch["wrist_angles"]

        # Generate predictions
        preds = self.forward(emg)

        # Slice the raw wrist angles to align with the network predictions
        wrist_angles = wrist_angles[
            :, :, self.network.left_context :: self.network.stride
        ]

        # Take one-step differences of wrist angles to get (rescaled) velocity labels.
        # We do this after slicing so that the differences are taken at the model
        # output frequency (50Hz), not the frequency of the raw EMG (2000Hz). Since
        # we don't know the label for the first timestep, remove the first prediction.
        preds = preds[:, :, 1:]
        labels = torch.diff(wrist_angles, dim=2)

        # Compute loss
        loss = self.loss_fn(preds, labels)
        self.log(f"{stage}_loss", loss, sync_dist=True)

        # Log mean absolute error in degrees/second
        # The loss is in radians, so we convert to degrees and
        # multiply by the model output frequency (50Hz)
        mae_deg_s = np.rad2deg(loss.item()) * 50
        self.log(f"{stage}_mae_deg_per_sec", mae_deg_s, sync_dist=True)

        return loss


class FingerStateMaskGenerator(torch.nn.Module):
    """
    Generate finger state masks based on press and release event labels.

    Input labels are tensors of shape [batch, num_gestures, times] where each channel has pulses
    representing events. Each pulse has a duration of 40ms (8 samples at 200 Hz).

    The output mask has value 1 from the beginning of the press until the end of the release,
    and 0 elsewhere for each finger.

    Parameters
    ----------
    lpad : int
        Time bin padding before the press event
    rpad : int
        Time bin padding after the release event
    """

    def __init__(
        self,
        lpad: int = 0,
        rpad: int = 0,
    ) -> None:
        super().__init__()

        self.lpad = lpad
        self.rpad = rpad

        # Define finger output channels
        self.INDEX_FINGER = 0
        self.MIDDLE_FINGER = 1

    def forward(self, gesture_labels: torch.Tensor) -> torch.Tensor:
        """
        Generate finger state masks from gesture labels using diff to find event onsets

        Parameters
        ----------
        gesture_labels : torch.Tensor
            Tensor of shape [batch, num_gestures, times] where each channel corresponds to gesture types
            defined in GestureType.

            Each gesture is represented as a pulse with 40ms duration (8 samples at 200 Hz)

        Returns
        -------
        torch.Tensor
            Tensor of shape [batch, 2, times] where each channel corresponds to:
                0: index finger state (1 when pressed, 0 when released)
                1: middle finger state (1 when pressed, 0 when released)
        """
        batch_size, _, time_steps = gesture_labels.shape

        # Initialize output masks for both fingers
        finger_masks = torch.zeros(
            (batch_size, 2, time_steps),
            device=gesture_labels.device,
            dtype=torch.float32,
        )

        # Process each sequence in the batch
        for b in range(batch_size):
            # Process index finger
            self._process_finger(
                gesture_labels[b],
                finger_masks[b],
                press_channel=GestureType.index_press.value,
                release_channel=GestureType.index_release.value,
                output_channel=self.INDEX_FINGER,
                time_steps=time_steps,
            )

            # Process middle finger
            self._process_finger(
                gesture_labels[b],
                finger_masks[b],
                press_channel=GestureType.middle_press.value,
                release_channel=GestureType.middle_release.value,
                output_channel=self.MIDDLE_FINGER,
                time_steps=time_steps,
            )

        return finger_masks

    def _process_finger(
        self,
        gesture_labels: torch.Tensor,
        finger_masks: torch.Tensor,
        press_channel: int,
        release_channel: int,
        output_channel: int,
        time_steps: int,
    ) -> None:
        """
        Process a single finger's events to create its state mask

        Parameters
        ----------
        gesture_labels : torch.Tensor
            Gesture labels for a single batch item [9, times]
        finger_masks : torch.Tensor
            Output mask tensor for a single batch item [2, times]
        press_channel : int
            Channel index for press events
        release_channel : int
            Channel index for release events
        output_channel : int
            Output channel index
        time_steps : int
            Total number of time steps
        """
        # Extract press and release signals for this finger
        press_signal = gesture_labels[press_channel]
        release_signal = gesture_labels[release_channel]

        # Calculate diff to find onsets, adding a zero at the beginning to maintain size
        zero_tensor = torch.zeros(1, device=gesture_labels.device)
        press_diff = torch.diff(press_signal, n=1, prepend=zero_tensor)
        release_diff = torch.diff(release_signal, n=1, prepend=zero_tensor)

        # Find indices where diff > 0 (onset detection)
        press_onsets = torch.nonzero(press_diff > 0, as_tuple=True)[0]
        release_onsets = torch.nonzero(release_diff > 0, as_tuple=True)[0]

        # Ensure we have both press and release events
        if press_onsets.numel() == 0 or release_onsets.numel() == 0:
            return

        # For each press, find the next release
        for press_idx in press_onsets:
            # Find all releases that occur after this press
            future_releases = release_onsets[release_onsets > press_idx]

            # If there's no future release, use the end of the sequence
            if future_releases.numel() == 0:
                release_idx = torch.tensor(time_steps - 1, device=finger_masks.device)
            else:
                # Use the first future release
                release_idx = future_releases[0]

            # Apply padding (with bounds checking)
            start_idx = torch.clamp(press_idx - self.lpad, min=0)
            end_idx = torch.clamp(release_idx + self.rpad + 1, max=time_steps)

            # Set mask to 1 between press and release (inclusive)
            finger_masks[output_channel, start_idx:end_idx] = 1.0


class DiscreteGesturesModule(BaseLightningModule):
    def __init__(
        self,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        weight_decay: float,
        lr_scheduler_patience: int,
        lr_scheduler_factor: float,
        lr_scheduler_min_lr: float,
        monitor_metric: str,
    ) -> None:

        super().__init__(network=network, optimizer=optimizer)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.mask_generator = FingerStateMaskGenerator(lpad=0, rpad=7)

    def _step(self, batch: Mapping[str, torch.Tensor], stage: str = "train") -> float:

        # Extract data
        emg = batch["emg"]
        targets = batch["targets"]
        targets = targets[:, :, self.network.left_context :: self.network.stride]
        release_mask = self.mask_generator(targets)
        mask = torch.ones_like(targets)
        mask[
            :, [GestureType.index_release.value, GestureType.middle_release.value], :
        ] = release_mask

        # Generate predictions
        preds = self.forward(emg)

        # Compute loss
        loss = self.loss_fn(preds, targets)
        loss = (loss * mask).sum() / mask.sum()
        self.log(f"{stage}_loss", loss, sync_dist=True)

        if stage == "test":
            prompts = batch["prompts"][0]
            times = batch["timestamps"][0]
            preds = nn.Sigmoid()(preds)
            preds = preds.squeeze(0).detach().cpu().numpy()
            times = times[self.network.left_context :: self.network.stride]
            cler = compute_cler(preds, times, prompts)
            self.log("test_cler", cler, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer with learning rate scheduler that decays on plateau."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.hparams.lr_scheduler_factor,
                patience=self.hparams.lr_scheduler_patience,
                min_lr=self.hparams.lr_scheduler_min_lr,
                verbose=True,
            ),
            "monitor": self.hparams.monitor_metric,
            "interval": "epoch",
            "frequency": 1,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


class HandwritingModule(BaseLightningModule):
    def __init__(
        self,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: dict,
        decoder: Decoder,
    ) -> None:
        super().__init__(network=network, optimizer=optimizer)

        self.lr_scheduler = lr_scheduler

        # Criterion
        self.ctc_loss = nn.CTCLoss(
            blank=charset().null_class,
            zero_infinity=True,
        )

        # Decoder
        self.decoder = decoder

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )
        torch.autograd.set_detect_anomaly(True)

    def _step(self, batch: Mapping[str, torch.Tensor], stage: str = "train") -> float:
        emg = batch["emg"]
        prompts = batch["prompts"]

        emg_lengths = batch["emg_lengths"]
        target_lengths = batch["prompt_lengths"]
        N = len(emg_lengths)  # batch_size

        emissions, slc = self.forward(emg)
        emission_lengths = self.network.compute_time_downsampling(
            emg_lengths=emg_lengths, slc=slc  # (N,)
        )
        loss = self.ctc_loss(
            log_probs=emissions.movedim(
                0, 1
            ),  # (N,T,num_classes) -> (T, N, num_classes)
            targets=prompts,  # (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )
        self.log(f"{stage}_loss", loss, sync_dist=True)

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.movedim(0, 1)
            .detach()
            .cpu()
            .numpy(),  # (T, N, num_classes) -> (T, N, num_classes)
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{stage}_metrics"]
        prompts = prompts.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad prompts (T, N) for batch entry
            target = prompts[i, : target_lengths[i]]
            metrics.update(
                prediction=self.decoder._charset.labels_to_str(predictions[i]),
                target=self.decoder._charset.labels_to_str(target),
            )

            if i == N - 1:
                print(
                    f"pred: {self.decoder._charset.labels_to_str(predictions[i])}, target: {self.decoder._charset.labels_to_str(target)}"
                )

        return loss

    def on_train_epoch_end(self) -> None:
        self._on_epoch_end(stage="train")

    def on_validation_epoch_end(self) -> None:
        self._on_epoch_end(stage="val")

    def on_test_epoch_end(self) -> None:
        self._on_epoch_end(stage="test")

    def _on_epoch_end(self, stage: str) -> None:
        metrics = self.metrics[f"{stage}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def configure_optimizers(self):
        self.optimizer = self.optimizer(self.parameters())
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[
                        s(self.optimizer) for s in self.lr_scheduler["schedules"]
                    ],
                    milestones=self.lr_scheduler["milestones"],
                ),
                "interval": self.lr_scheduler["interval"],
            },
        }
