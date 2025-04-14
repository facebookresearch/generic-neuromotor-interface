# pyre-strict
import h5py
import numpy as np
import pandas as pd


class EMGData:
    """
    Light wrapper around EMG hdf5 files.
    Used for basic data exploration in explore_data.ipynb.
    Not used in model training.
    """

    def __init__(self, hdf5_path: str) -> None:
        self.hdf5_path = hdf5_path

        with h5py.File(self.hdf5_path, "r") as file:
            self.timeseries = file["data"][:]
            self.task = file["data"].attrs["task"]

        self.stages = pd.read_hdf(hdf5_path, "stages")

    def partition(self, start_t: float, end_t: float) -> np.ndarray:
        """Slice timeseries data between the given timestamps."""
        start_idx, end_idx = self.time.searchsorted([start_t, end_t])
        return self.timeseries[start_idx:end_idx]

    @property
    def emg(self) -> np.ndarray:
        """Shape (time, chanel), units volts."""
        return self.timeseries["emg"]

    @property
    def time(self) -> np.ndarray:
        """Shape (time,), units seconds."""
        return self.timeseries["time"]


class DiscreteGesturesData(EMGData):
    def __init__(self, hdf5_path: str) -> None:
        super().__init__(hdf5_path)
        assert self.task == "discrete_gestures"
        self.prompts = pd.read_hdf(hdf5_path, "prompts")


class HandwritingData(EMGData):
    def __init__(self, hdf5_path: str) -> None:
        super().__init__(hdf5_path)
        assert self.task == "handwriting"
        self.prompts = pd.read_hdf(hdf5_path, "prompts")


class WristAngleData(EMGData):
    def __init__(self, hdf5_path: str) -> None:
        super().__init__(hdf5_path)
        assert self.task == "wrist"

    @property
    def wrist_angles(self) -> np.ndarray:
        """
        Shape (time, channel), units radians.

        First channel is extension/flexion, where positive values
        correspond to extension and negative values to flexion.

        Second channel is radial/ulnar deviation, where positive values
        correspond to radial deviation, and negative values to ulnar deviation.
        """
        return self.timeseries["wrist_angles"]


LOADERS = {
    "discrete_gestures": DiscreteGesturesData,
    "wrist": WristAngleData,
    "handwriting": HandwritingData,
}


def load_data(hdf5_path: str) -> EMGData:
    """Load a dataset, automatically determining the correct loader
    for the dataset type."""
    with h5py.File(hdf5_path, "r") as file:
        task = file["data"].attrs["task"]
    return LOADERS[task](hdf5_path)
