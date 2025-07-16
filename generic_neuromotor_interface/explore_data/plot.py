# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import numpy as np


def plot_emg(
    time: np.ndarray,
    emg: np.ndarray,
    vertical_offset_quantile: float = 0.999,
    scale_bar_mv: float = 1,
    normalize_time: bool = True,
    ax: plt.Axes | None = None,
) -> None:
    """Plot EMG over time, with each channel vertically offset.

    Parameters
    ----------
    time : np.ndarray, shape (time,)
        Time array (seconds).
    emg : np.ndarray, shape (time, channel)
        EMG array (volts).
    vertical_offset_quantile : float, optional
        The quantile to use for the vertical offset. Larger values result
        in greater vertical separation between channels.
    scale_bar_mv : float, optional
        The value of the vertical scale bar (in mV).
    normalize_time : bool, optional
        If True, subtract the first time value from all time values.
    ax : plt.Axes, optional
        Axes to plot on. If None, a new figure will be created.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    if normalize_time:
        time = time.copy() - time[0]

    num_channels = emg.shape[1]
    vertical_offset = np.quantile(np.abs(emg), vertical_offset_quantile) * 2
    vertical_offsets = np.arange(num_channels) * vertical_offset
    ax.plot(time, emg + vertical_offsets)

    # Add vertical scale bar
    dy = -vertical_offset / 2
    ax.vlines(
        time[0],
        dy,
        dy + scale_bar_mv / 1000,
        color="k",
        linewidth=3,
    )
    ax.text(
        time[0],
        dy + scale_bar_mv / 2000,
        f"{scale_bar_mv} mV",
        rotation=90,
        ha="right",
        va="center",
        fontsize=8,
    )

    # Format axes
    ax.set(
        xlabel="Time (seconds)",
        ylabel="EMG",
        xlim=[time[0], time[-1]],
        yticks=[],
        ylim=[-vertical_offset, vertical_offset * num_channels],
    )
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_wrist(
    time: np.ndarray,
    wrist: np.ndarray,
    normalize_time: bool = True,
    ax: plt.Axes | None = None,
) -> None:
    """
    Plot wrist angles over time.

    Parameters
    ----------
    time : np.ndarray, shape (time,)
        Time array (seconds).
    wrist : np.ndarray, shape (time, channel)
        Wrist angles array (radians).
        First channel is flexion/extension, second is radial/ulnar deviation.
    normalize_time : bool, optional
        If True, subtract the first time value from all time values.
    ax : plt.Axes, optional
        Axes to plot on. If None, a new figure will be created.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    if normalize_time:
        time = time.copy() - time[0]

    ax.plot(time, wrist)

    ax.set(
        xlabel="Time (seconds)",
        ylabel="Wrist angle\n(radians)",
        xlim=[time[0], time[-1]],
    )

    ax.legend(["Flexion/extension", "Radial/ulnar deviation"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
