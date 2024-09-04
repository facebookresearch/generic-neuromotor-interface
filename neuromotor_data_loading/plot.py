import matplotlib.pyplot as plt
import numpy as np


def plot_wrist(
    time: np.ndarray,  # (time,)
    wrist: np.ndarray,  # (time, channel)
    normalize_time: bool = True,
    ax: plt.Axes | None = None,
) -> None:

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    if normalize_time:
        time = time.copy() - time[0]

    num_channels = wrist.shape[1]
    ax.plot(time, wrist)

    ax.set(
        xlabel="Time (seconds)",
        ylabel="Wrist angle\n(degrees)",
        xlim=[time[0], time[-1]],
    )


def plot_emg(
    time: np.ndarray,  # (time,)
    emg: np.ndarray,  # (time, channel)
    vertical_offset_quantile: float = 0.9999,
    normalize_time: bool = True,
    ax: plt.Axes | None = None,
) -> None:

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    if normalize_time:
        time = time.copy() - time[0]

    num_channels = emg.shape[1]
    vertical_offset = np.quantile(np.abs(emg), vertical_offset_quantile) * 2
    vertical_offsets = np.arange(num_channels) * vertical_offset
    ax.plot(time, emg + vertical_offsets)

    yticklabels = np.arange(num_channels) + 1
    ax.set(
        xlabel="Time (seconds)",
        ylabel="EMG (a.u.)",
        xlim=[time[0], time[-1]],
        yticks=vertical_offsets,
        yticklabels=yticklabels,
        ylim=[-vertical_offset, vertical_offset * num_channels],
    )
