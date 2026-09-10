import numpy as np
import torch
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from torch.utils.data import DataLoader, TensorDataset

from config.config_mcad_parameters import OUTPUT_NEURON_LIST, INPUT_NEURON_LIST
from modules.data_managment.csv_io import import_data_from_csv
from config.config_paths import *


class DataTensor:

    def __init__(
            self,
            filename: str,
            label_axis: tuple[str, str, str] = ("RMSPhaseCurrent_Q", "RMSPhaseCurrent_D", "ShaftSpeed"),
            batch_size: int = 10
    ):
        self.data_df = self._get_data(filename)
        self.num_machines = 0
        self.num_mappoints_per_dim = 0
        self.label_axis = label_axis
        self.machine_geomtry, self.map_tensor = self._create_map_tensor()
        geometry_tensor_train, geomtry_tensor_validation, map_tensor_train, map_tensor_validation = self._split_data()

        self.dataloader_train = DataLoader(
            TensorDataset(geometry_tensor_train, map_tensor_train),
            batch_size=batch_size,
            shuffle=True
        )
        self.dataloader_validation = DataLoader(
            TensorDataset(geomtry_tensor_validation, map_tensor_validation),
            batch_size=batch_size,
            shuffle=False
        )


    def _create_map_tensor(self) -> tuple[torch.Tensor, torch.Tensor]:
        machines = self._identify_machines_in_csv()
        axis = self._identify_map_axis_values()
        axis_indicies = self._create_axis_indicies(axis)
        x = self._create_nan_array()

        machine_geometry = []
        for idx_machine, (geometry_tuple, map_df) in enumerate(machines):
            machine_geometry.append(geometry_tuple)
            for _, row in map_df.iterrows():
                map_indicies = [
                    axis_indicies[axe][row[label]] for axe, label in enumerate(self.label_axis)
                ] #list with three indicies: index on the axe "axe" of the value row[label]

                for idx_channel, channel in enumerate(OUTPUT_NEURON_LIST):
                    x[idx_machine, idx_channel, map_indicies[0], map_indicies[1], map_indicies[2]] = row[channel]

        machine_geometry = torch.tensor(
            machine_geometry,
            dtype=torch.float32
        )

        return machine_geometry, torch.tensor(x, dtype=torch.float32)


    @staticmethod
    def _get_data(filename: str) -> pd.DataFrame:
        data_df = import_data_from_csv(
            PATH_DATA_MCAD_RESULTS,
            filename=filename
        )
        return data_df


    def _identify_machines_in_csv(self) -> DataFrameGroupBy:

        machines = self.data_df.groupby(
            INPUT_NEURON_LIST,
            sort=False
        )
        self.num_machines = len(machines)
        print(f"Found {len(machines)} machines")
        return machines


    def _identify_map_axis_values(self)-> list[list]:

        axis = []
        for i in range(len(self.label_axis)):
            axis.append(sorted(self.data_df[self.label_axis[i]].unique()))

        if len(axis[0]) != len(axis[1]) != len(axis[2]):
            raise ValueError(
                f"Not all axis have the same number of values:\n"
                f" x axis: {len(axis[0])}\n"
                f" y axis: {len(axis[1])}\n"
                f" z axies: {len(axis[2])}"
            )
        else:
            self.num_mappoints_per_dim = len(axis[0])

        return axis


    @staticmethod
    def _create_axis_indicies(three_axis: list[list]) -> list[dict]:

        axis_indicies = []
        for axis in three_axis:
            idx = {
                value: idx
                for idx, value in enumerate(axis)
            }
            axis_indicies.append(idx)

        return axis_indicies


    def _create_nan_array(self) -> np.ndarray:

        if self.num_machines == 0:
            raise ValueError("No Machines found")
        if self.num_mappoints_per_dim == 0:
            raise ValueError("No Mappoints found")

        return np.full(
            (
                self.num_machines,
                len(OUTPUT_NEURON_LIST),
                self.num_mappoints_per_dim,
                self.num_mappoints_per_dim,
                self.num_mappoints_per_dim
            ),
            np.nan,
            dtype=np.float32
        )


    def _split_data(self, test_split: float = 0.2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        num_validation = int(self.num_machines * test_split)
        num_training = self.num_machines - num_validation

        generator = torch.Generator().manual_seed(42)

        indices = torch.randperm(
            self.num_machines,
            generator=generator
        )

        train_indices = indices[:num_training]
        validation_indices = indices[num_training:]

        if self.map_tensor is not None and self.machine_geomtry is not None:
            x_train = self.machine_geomtry[train_indices]
            x_validation = self.machine_geomtry[validation_indices]
            y_train = self.map_tensor[train_indices]
            y_validation = self.map_tensor[validation_indices]
        else:
            raise RuntimeError("Tensor is None")

        return x_train, x_validation, y_train, y_validation
