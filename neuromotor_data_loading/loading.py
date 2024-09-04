import h5py
import numpy as np
import pandas as pd


class EMGData:
    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        self.data, self.task = self.load_data()

    def load_data(self):
        with h5py.File(self.hdf5_path, "r") as file:
            data = file["data"][:]
            task = file["data"].attrs["task"]
        return data, task


class DiscreteGesturesData(EMGData):
    def __init__(self, hdf5_path: str):
        super().__init__(hdf5_path)
        assert self.task == "discrete_gestures"

        self.emg = self.data["emg"]
        self.time = self.data["time"]
        self.labels = pd.read_hdf(hdf5_path, "labels")


class HandwritingData(EMGData):
    def __init__(self, hdf5_path: str):
        super().__init__(hdf5_path)
        assert self.task == "handwriting"

        self.emg = self.data["emg"]
        self.time = self.data["time"]
        self.labels = pd.read_hdf(hdf5_path, "labels")


class WristPoseData(EMGData):
    def __init__(self, hdf5_path: str):
        super().__init__(hdf5_path)
        assert self.task == "wrist_pose"

        self.emg = self.data["emg"]
        self.wrist_angles = self.data["wrist_angles"]
        self.time = self.data["time"]


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
