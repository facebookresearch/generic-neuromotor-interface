# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from generic_neuromotor_interface.constants import GestureType

from generic_neuromotor_interface.handwriting_utils import charset

# Transforms are applied to the EMG timeseries and prompts dataframe.
Transform = Callable[[np.ndarray, pd.DataFrame | None], torch.Tensor]


def _to_tensor(data: np.ndarray | list[Any]) -> torch.Tensor:
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

    # Pulse extends from (time + pulse_window[0] to time + pulse_window[1])
    pulse_window: list[float]  # Seconds

    def __call__(
        self, timeseries: np.ndarray, prompts: pd.DataFrame | None
    ) -> torch.Tensor:
        assert prompts is not None

        # Get gesture prompts within the timeseries window
        tlim = (timeseries["time"][0], timeseries["time"][-1])
        prompts = prompts[prompts["time"].between(*tlim)]
        prompts = prompts[
            prompts["name"].isin([gesture.name for gesture in GestureType])
        ]

        # Convert to binary pulse matrix
        targets = self.gesture_times_to_targets(
            timeseries["time"],
            prompts["time"],
            prompts["name"].map(
                {gesture.name: gesture.value for gesture in GestureType}
            ),
        )

        return {
            "emg": _to_tensor(timeseries["emg"].T),
            "targets": targets,
        }

    def gesture_times_to_targets(
        self,
        times: np.ndarray,
        event_start_times: np.ndarray,
        event_ids: pd.Series,
    ) -> torch.Tensor:
        """
        Convert gesture times to a (num_events, time) binary pulse matrix with 1.0 for
        the duration of the event.
        """

        assert len(event_start_times) == len(event_ids)

        num_timesteps = len(times)
        duration = times[-1] - times[0]
        sampling_freq = int(num_timesteps / duration)

        event_ids = event_ids.to_numpy()

        # Indices of each event in the pulse matrix
        event_time_indices = np.searchsorted(times, event_start_times)
        pulse = torch.zeros(len(GestureType), num_timesteps, dtype=torch.float32)

        valid_events = (event_time_indices > 0) & (event_time_indices < num_timesteps)
        valid_indices = np.where(valid_events)[0]

        for idx in valid_indices:
            event_start = event_time_indices[idx]
            event_id = event_ids[idx]

            if event_id >= len(GestureType) or event_id < 0:
                continue

            start_offset = int(self.pulse_window[0] * sampling_freq)
            end_offset = int(self.pulse_window[1] * sampling_freq)

            start_idx = max(0, event_start + start_offset)
            end_idx = min(num_timesteps, event_start + end_offset)

            if start_idx < end_idx:
                pulse[event_id, start_idx:end_idx] = 1.0

        return pulse


@dataclass
class HandwritingTransform:
    """Extract emg and prompts."""

    def __init__(self) -> None:
        self.charset = charset()

    def __call__(
        self, timeseries: np.ndarray, prompt: str | None
    ) -> dict[str, torch.Tensor | str]:
        assert prompt is not None
        return {
            "emg": _to_tensor(timeseries["emg"]),  # (T, C)
            "prompts": _to_tensor(
                self.charset.str_to_labels(prompt)
            ).long(),  # (sequence_length)
        }
