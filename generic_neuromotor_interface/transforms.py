# pyre-strict
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
import torch

from generic_neuromotor_interface.handwriting_utils import charset

# Transforms are applied to the EMG timeseries and prompts dataframe.
Transform = Callable[[np.ndarray, pd.DataFrame | None], torch.Tensor]


def _to_tensor(data: np.ndarray) -> torch.Tensor:

    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float()
    elif isinstance(data, list):
        return torch.tensor(data).float()
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


@dataclass
class WristTransform:
    """Extract EMG and wrist angles."""

    def __call__(
        self, timeseries: np.ndarray, prompts: pd.DataFrame | None
    ) -> torch.Tensor:
        emg = _to_tensor(timeseries["emg"])
        wrist_angles = _to_tensor(timeseries["wrist_angles"])

        # Get only the indices we want to train our model on
        # We train on only the first dimension of wrist angles,
        # corresponding to the wrist flexion/extension angle.
        wrist_angles = wrist_angles[:, [0]]

        # Reshape data from (time, channel) to (channel, time)
        emg = emg.permute(1, 0)
        wrist_angles = wrist_angles.permute(1, 0)

        return {"emg": emg, "wrist_angles": wrist_angles}


@dataclass
class DiscreteGesturesTransform:
    """
    Extract EMG and discrete gesture times.
    Convolve gesture times with a step function to create targets.
    """

    pulse_durations: dict[str, tuple[float, float]]

    @cached_property
    def gestures(self):
        return list(self.pulse_durations.keys())

    @cached_property
    def gesture_to_index(self):
        return {gesture: i for i, gesture in enumerate(self.gestures)}

    def __call__(
        self, timeseries: np.ndarray, prompts: pd.DataFrame | None
    ) -> torch.Tensor:

        # Get gesture prompts within the timeseries window
        tlim = (timeseries["time"][0], timeseries["time"][-1])
        prompts = prompts[prompts["time"].between(*tlim)]
        prompts = prompts[prompts["name"].isin(self.gestures)]

        # Convert to binary pulse matrix
        targets = self.gesture_times_to_targets(
            timeseries["time"],
            prompts["time"],
            prompts["name"].map(self.gesture_to_index),
        )

        return {
            "emg": _to_tensor(timeseries["emg"].T),
            "targets": targets,
        }

    def gesture_times_to_targets(
        self,
        times: np.ndarray,
        event_start_times: np.ndarray,
        event_ids: list[int],
    ) -> torch.Tensor:
        """
        Convert gesture times to a (num_events, time) binary pulse matrix with 1.0 for
        the duration of the event.
        """

        assert len(event_start_times) == len(event_ids)

        num_timesteps = len(times)
        duration = times[-1] - times[0]
        sampling_freq = int(num_timesteps / duration)

        # Indices of each event in the pulse matrix
        event_time_indices = np.searchsorted(times, event_start_times)
        pulse = torch.zeros(len(self.gestures), num_timesteps, dtype=torch.float32)

        for event_start, event_id in zip(event_time_indices, event_ids):
            event = self.gestures[event_id]
            start = event_start + self.pulse_durations[event][0] * sampling_freq
            end = event_start + self.pulse_durations[event][1] * sampling_freq
            pulse[event_id, int(start) : int(end)] = 1.0

        return pulse


@dataclass
class HandwritingTransform:
    """Extract emg and prompts."""

    def __init__(self):
        self.charset = charset()

    def __call__(
        self, timeseries: np.ndarray, prompt: str | None
    ) -> dict[str, torch.Tensor | str]:
        return {
            "emg": _to_tensor(timeseries["emg"]),  # (T, C)
            "prompts": _to_tensor(
                self.charset.str_to_labels(prompt)
            ).long(),  # (sequence_length)
        }
