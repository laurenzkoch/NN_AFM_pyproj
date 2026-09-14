
import math
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
            label_axis: tuple[str, str, str] = ("RMSPhaseCurrent_D_density", "RMSPhaseCurrent_Q_density", "ShaftSpeed"),
            batch_size: int = 10
    ) -> None:

        self.data_df = self._get_data(filename)
        self._recalculate_data()

        self.label_axis = label_axis
        self.axis: list[list] = self._identify_map_axis_values()
        self.map_shape = tuple(len(axis) for axis in self.axis)
        self.axis_indices: list[dict] = self._create_axis_indices()

        self._clean_data()

        self.machines: DataFrameGroupBy = self._identify_machines_in_csv()
        self.num_machines = len(self.machines)

        self.machine_geometry_tensor = self._create_machine_geometry_tensor()
        self.nan_array: np.ndarray = self._create_nan_array()
        self.map_tensor: torch.Tensor = self._create_map_tensor()

        (
            x_train,
            x_validation,
            y_train,
            y_validation
        ) = self._split_data(test_split=0.2)

        self.dataloader_train = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=True
        )

        self.dataloader_validation = DataLoader(
            TensorDataset(x_validation, y_validation),
            batch_size=batch_size,
            shuffle=False
        )


    @staticmethod
    def _get_data(filename: str) -> pd.DataFrame:
        data_df = import_data_from_csv(
            PATH_DATA_MCAD_RESULTS,
            filename=filename
        )
        return data_df

    def _recalculate_data(self) -> None:
        conductor_area = (
                self.data_df["PeakCurrent"]
                / (np.sqrt(2)**2 * self.data_df["RMSCurrentDensity"])
        )
        self.data_df["RMSPhaseCurrent_D_density"] = round(self.data_df["RMSPhaseCurrent_D"]/ conductor_area, 2)
        self.data_df["RMSPhaseCurrent_Q_density"] = round(self.data_df["RMSPhaseCurrent_Q"]/ conductor_area, 2)


    def _clean_data(self) -> None:

        _machines = self._identify_machines_in_csv()
        points_per_map = math.prod(self.map_shape)
        _indices_to_delete = []

        maps_with_duplicate_points = 0
        uncomplete_maps = 0

        for idx_machine, (geometry_tuple, map_df) in enumerate(_machines):

            operating_points = map_df[list(self.label_axis)]

            # machine multiple times in csv
            if operating_points.duplicated().any():
                _indices_to_delete.extend(map_df.index)
                maps_with_duplicate_points += 1
                continue

            # map not complete
            if len(map_df) != points_per_map:
                _indices_to_delete.extend(map_df.index)
                uncomplete_maps += 1

        self.data_df = self.data_df.drop(_indices_to_delete)

        print(
            f"uncomplete maps: {uncomplete_maps}\n"
            f"machines multiple times in csv: {maps_with_duplicate_points}\n"
        )


    def _identify_machines_in_csv(self) -> DataFrameGroupBy:

        _machines = self.data_df.groupby(
            INPUT_NEURON_LIST,
            sort=False
        )
        print(f"Found {len(_machines)} machines")
        return _machines


    def _identify_map_axis_values(self)-> list[list]:

        _axis = []
        for i in range(len(self.label_axis)):
            _axis.append(sorted(self.data_df[self.label_axis[i]].unique()))
        return _axis


    def _create_axis_indices(self) -> list[dict]:

        _axis_indices = []
        for _axis in self.axis:
            idx = {
                value: idx
                for idx, value in enumerate(_axis)
            }
            _axis_indices.append(idx)
        return _axis_indices


    def _create_machine_geometry_tensor(self) -> torch.Tensor:

        _machine_geometry = []
        for idx_machine, (geometry_tuple, map_df) in enumerate(self.machines):
            _machine_geometry.append(geometry_tuple)
        return torch.tensor(
                _machine_geometry,
                dtype=torch.float32
            )


    def _create_nan_array(self) -> np.ndarray:

        return np.full(
            (
                self.num_machines,
                len(OUTPUT_NEURON_LIST),
                *self.map_shape
            ),
            np.nan,
            dtype=np.float32
        )


    def _create_map_tensor(self) -> torch.Tensor:
        #go throgh each machine
        for idx_machine, (geometry_tuple, map_df) in enumerate(self.machines):

            #go through each op
            for _, row in map_df.iterrows():
                map_indices = [
                    self.axis_indices[axe][row[label]] for axe, label in enumerate(self.label_axis)
                ] #list with three indices: index on the axe "axe" of the value row[label]

                #go through each channel for op
                for idx_channel, channel in enumerate(OUTPUT_NEURON_LIST):
                    self.nan_array[idx_machine, idx_channel, map_indices[0], map_indices[1], map_indices[2]] = row[channel]

        _map_tensor = torch.tensor(self.nan_array, dtype=torch.float32)

        nan_indices = torch.isnan(_map_tensor).nonzero(as_tuple=False)
        num_nan = nan_indices.size(0)
        if num_nan > 0:
            raise ValueError(
                f"Map tensor contains {num_nan} NaN values\n"
                f"Index order (zero-based): (machine, channel, {', '.join(self.label_axis)})\n"
                f"NaN indices: {nan_indices.tolist()}"
            )
        return _map_tensor


    def _split_data(self, test_split: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.num_machines < 5:
            raise ValueError(
                f"At least 5 complete machines required, got {self.num_machines}"
            )

        num_validation = int(self.num_machines * test_split)
        num_training = self.num_machines - num_validation

        generator = torch.Generator().manual_seed(42)

        indices = torch.randperm(
            self.num_machines,
            generator=generator
        )

        train_indices = indices[:num_training]
        validation_indices = indices[num_training:]

        x_train = self.machine_geometry_tensor[train_indices]
        x_validation = self.machine_geometry_tensor[validation_indices]
        y_train = self.map_tensor[train_indices]
        y_validation = self.map_tensor[validation_indices]

        return x_train, x_validation, y_train, y_validation
