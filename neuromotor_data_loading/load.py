import h5py
import numpy as np
import pandas as pd


class EMGData:
    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        self.timeseries, self.task = self.load_data()

    def load_data(self):
        with h5py.File(self.hdf5_path, "r") as file:
            timeseries = file["data"][:]
            task = file["data"].attrs["task"]
        return timeseries, task

    def partition(self, start_t: float, end_t: float) -> np.ndarray:
        """Slice timeseries data between the given timestamps."""
        start_idx, end_idx = self.time.searchsorted([start_t, end_t])
        return self.timeseries[start_idx:end_idx]

    @property
    def emg(self):
        return self.timeseries["emg"]

    @property
    def time(self):
        return self.timeseries["time"]


class DiscreteGesturesData(EMGData):
    def __init__(self, hdf5_path: str):
        super().__init__(hdf5_path)
        assert self.task == "discrete_gestures"
        self.labels = pd.read_hdf(hdf5_path, "labels")


class HandwritingData(EMGData):
    def __init__(self, hdf5_path: str):
        super().__init__(hdf5_path)
        assert self.task == "handwriting"
        self.labels = pd.read_hdf(hdf5_path, "labels")


class WristPoseData(EMGData):
    def __init__(self, hdf5_path: str):
        super().__init__(hdf5_path)
        assert self.task == "wrist_pose"

    @property
    def wrist_angles(self):
        return self.timeseries["wrist_angles"]


LOADERS = {
    "discrete_gestures": DiscreteGesturesData,
    "wrist_pose": WristPoseData,
    "handwriting": HandwritingData,
}


def load_data(hdf5_path: str):
    """Load a dataset, automatically determining the correct loader for the dataset type."""
    with h5py.File(hdf5_path, "r") as file:
        task = file["data"].attrs["task"]
    return LOADERS[task](hdf5_path)
