import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pandas.core.groupby.generic import DataFrameGroupBy
from torch import Tensor

from src.config.config_mcad_parameters import CSV_PARAMETERS, AXIS_LIST, LAMBDA_LIST, MACHINE_DEFINITION, MAP_SIZE, \
    OUTPUT_NEURON_LIST, GEO_PARAMETERS, GEO_PARAMETERS_MIN_MAX_DICT
from src.modules.data_managment.csv_io import import_data_from_csv

logger = logging.getLogger(__name__)


@dataclass
class MachineDataTensor:
    geometry_tensor: Tensor
    map_tensor: Tensor

    def shuffle(self):
        generator = torch.Generator().manual_seed(42)
        permutation = torch.randperm(
            self.geometry_tensor.shape[0],
            generator=generator,
        )
        self.geometry_tensor = self.geometry_tensor[permutation]
        self.map_tensor = self.map_tensor[permutation]


@dataclass
class TrainValidationTestDataTensors:
    train: MachineDataTensor
    validation: MachineDataTensor
    test: MachineDataTensor | None


class DataLoader:

    def __init__(self, training_directory: Path) -> None:
        self.training_directory = training_directory

    def load(self, filename: str) -> TrainValidationTestDataTensors:

        def _func(_filename) -> MachineDataTensor:
            _dataframe: pd.DataFrame = import_data_from_csv(self.training_directory / _filename)
            self._test_columns(_dataframe)
            _dataframe = _dataframe[CSV_PARAMETERS]
            _dataframe = self._recalculate_data(_dataframe)
            _dataframe = self._clean_data(_dataframe)
            _machines_groups: DataFrameGroupBy = self._identify_machines_in_csv(_dataframe)
            _map_values_indices: dict[str, dict[float, int]] = self._get_map_axis_with_indinces(_dataframe)
            _num_machines: int = len(_machines_groups)
            _machines: MachineDataTensor = self._create_tensors(_machines_groups, _map_values_indices, _num_machines)
            _machines.geometry_tensor = self._normalize_geo_tensor(_machines.geometry_tensor)

            logger.info(f"Found: {len(_machines_groups)} machines in {filename}")

            return _machines

        data_list: list[MachineDataTensor] = [
            _func(filename)
        ]
        for _idx, csv_file in enumerate(self.training_directory.glob("*.csv")):
            if csv_file.name != filename:
                machines = _func(csv_file.name)
                data_list.append(machines)

        num_machines: int = data_list[0].geometry_tensor.shape[0]
        train_indices, test_indices = self._split_indices(num_machines, test_split=0.2)

        train_data: MachineDataTensor = MachineDataTensor(
            geometry_tensor=data_list[0].geometry_tensor[train_indices],
            map_tensor=data_list[0].map_tensor[train_indices],
        )
        validation_data: MachineDataTensor = MachineDataTensor(
            geometry_tensor=data_list[0].geometry_tensor[test_indices],
            map_tensor=data_list[0].map_tensor[test_indices],
        )

        x_min_max = self._calc_min_max(train_data.map_tensor)
        train_data.map_tensor = self._normalize_map_tensor(train_data.map_tensor, x_min_max)
        validation_data.map_tensor = self._normalize_map_tensor(validation_data.map_tensor, x_min_max)

        test_data: MachineDataTensor | None
        if len(data_list) > 1:
            test_data: MachineDataTensor = self._fusion_validationdata(data_list[1:])
            test_data.map_tensor = self._normalize_map_tensor(test_data.map_tensor, x_min_max)
        else:
            test_data = None

        data = TrainValidationTestDataTensors(
            train=train_data,
            validation=validation_data,
            test=test_data,
        )
        return data

    @staticmethod
    def _test_columns(df: pd.DataFrame) -> None:
        missing = [col for col in CSV_PARAMETERS if col not in df.columns]
        if missing:
            logger.error(f"Missing parameters: {missing}")
            raise ValueError(f"Missing Column: {missing}")

    @staticmethod
    def _recalculate_data(df: pd.DataFrame) -> pd.DataFrame:

        conductor_area = df["PeakCurrent"] / (np.sqrt(2) * df["RMSCurrentDensity"])
        # Auf das nächste Vielfache von 0,05 runden.
        df[AXIS_LIST[1]] = (df["RMSPhaseCurrent_D"] / conductor_area * 20).round() / 20
        df[AXIS_LIST[2]] = (df["RMSPhaseCurrent_Q"] / conductor_area * 20).round() / 20

        d_si = df["Stator_Bore"]
        d_sa = df["Stator_Lam_Dia"]
        radial_width = (d_sa - d_si) / 2

        df[LAMBDA_LIST[0]] = d_si / d_sa
        df[LAMBDA_LIST[1]] = df["Magnet_Arc_[ED]"] / 180
        df[LAMBDA_LIST[2]] = df["Airgap"] / df["Magnet_Length"]

        # b_S / tau_s, b_S / d_S, h_PM / ((D_Sa - D_Si) / 2)
        df[LAMBDA_LIST[3]] = df["Slot_Number"] * df["Slot_Width"] / (math.pi * d_si)
        df[LAMBDA_LIST[4]] = df["Slot_Width"] / df["Slot_Depth"]
        df[LAMBDA_LIST[5]] = df["Magnet_Thickness"] / radial_width

        # d_emb / (radial_width - h_PM)
        df[LAMBDA_LIST[6]] = df["Magnet_Embed_Depth"] / (radial_width - df["Magnet_Thickness"])
        # (l_S - d_S) / tau_p, with tau_p = pi * (D_Sa + D_Si) / (2 * p)
        df[LAMBDA_LIST[7]] = (
                2 * df["Pole_Number"] * (df["Stator_Lam_Length"] - df["Slot_Depth"])
                / (math.pi * (d_sa + d_si))
        )
        df[LAMBDA_LIST[8]] = df["Magnet_Length"] / df["Rotor_Lam_Length"]

        return df

    @staticmethod
    def _identify_machines_in_csv(df: pd.DataFrame) -> DataFrameGroupBy:
        _machines = df.groupby(
            MACHINE_DEFINITION,
            sort=False
        )
        return _machines

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:

        _machines = self._identify_machines_in_csv(df)
        points_per_map = math.prod(MAP_SIZE)
        _indices_to_delete = []

        maps_with_duplicate_points = 0
        uncomplete_maps = 0

        for idx_machine, (geometry_tuple, map_df) in enumerate(_machines):

            operating_points = map_df[AXIS_LIST]

            # machine multiple times in csv
            if operating_points.duplicated().any():
                _indices_to_delete.extend(map_df.index)
                maps_with_duplicate_points += 1
                continue

            # map not complete
            if len(map_df) != points_per_map:
                _indices_to_delete.extend(map_df.index)
                uncomplete_maps += 1

        df = df.drop(_indices_to_delete)

        logger.info(
            f"uncomplete maps: {uncomplete_maps}\n"
            f"machines multiple times in csv: {maps_with_duplicate_points}\n"
        )

        return df

    @staticmethod
    def _get_map_axis_with_indinces(df: pd.DataFrame) -> dict[str, dict[float | int, int]]:
        # dict with one dict for each axe. keys are the axe labels.
        # inner dict: keys: axe-values; values: idx of axe-value

        _axis = {
            axe_label: {
                axe_value: _idx_of_axe_value
                for _idx_of_axe_value, axe_value in enumerate(sorted(df[axe_label].unique()))
            }
            for axe_label in AXIS_LIST
        }
        return _axis

    @staticmethod
    def _create_tensors(
            machine_groups: DataFrameGroupBy,
            map_indices: dict[str, dict[float, int]],
            num_machines: int
    ) -> MachineDataTensor:

        def _get_map_coordinates(_row: pd.Series, idx_dict: dict[str, dict[float, int]]) -> tuple[int, ...]:
            coordinates = tuple[int, ...](
                idx_dict[axe_label][_row[axe_label]]
                for axe_label in AXIS_LIST
            )
            return coordinates

        _machine_geometry_array = np.full(
            (
                num_machines,
                len(MACHINE_DEFINITION),
            ),
            np.nan,
            dtype=np.float32
        )
        _map_array = np.full(
            (
                num_machines,
                *MAP_SIZE,
                len(OUTPUT_NEURON_LIST)
            ),
            np.nan,
            dtype=np.float32
        )
        for idx_machine, (geometry_tuple, map_df) in enumerate(machine_groups):
            _machine_geometry_array[idx_machine, ...] = geometry_tuple

            for _idx, row in map_df.iterrows():
                _map_array[
                    idx_machine,
                    *_get_map_coordinates(row, map_indices),
                    :,
                ] = row[OUTPUT_NEURON_LIST].to_numpy(dtype=np.float32)

        nan_indices = np.argwhere(np.isnan(_map_array))
        num_nan = nan_indices.shape[0]
        if num_nan > 0:
            raise ValueError(
                f"Map tensor contains {num_nan} NaN values\n"
                f"Index order (zero-based): (machine,  {', '.join(AXIS_LIST)}, channel)\n"
                f"NaN indices: {nan_indices.tolist()}"
            )

        machines: MachineDataTensor = MachineDataTensor(
            geometry_tensor=torch.tensor(_machine_geometry_array, dtype=torch.float32),
            map_tensor=torch.tensor(_map_array, dtype=torch.float32),
        )

        return machines

    @staticmethod
    def _normalize_geo_tensor(geometry: Tensor) -> Tensor:
        for _idx, column_name in enumerate(GEO_PARAMETERS):
            x_min, x_max = GEO_PARAMETERS_MIN_MAX_DICT[column_name]
            denom = max(x_max - x_min, 1e-8)
            geometry[..., _idx] = (geometry[..., _idx] - x_min) / denom
        return geometry

    @staticmethod
    def _calc_min_max(tensor: Tensor) -> tuple[Tensor, Tensor]:
        x_min = tensor.amin(dim=(0, 1, 2, 3))
        x_max = tensor.amax(dim=(0, 1, 2, 3))
        return x_min, x_max

    @staticmethod
    def _normalize_map_tensor(_map: Tensor, x_min_max: tuple[Tensor, Tensor]) -> Tensor:
        x_min, x_max = x_min_max
        denom = torch.clamp(x_max - x_min, min=1e-8)
        map_norm = (_map - x_min) / denom
        return map_norm

    @staticmethod
    def _split_indices(
            num_machines: int,
            test_split: float,
    ) -> tuple[Tensor, Tensor]:

        if num_machines < 5:
            raise ValueError(
                f"At least 5 complete machines required, got {num_machines}"
            )

        num_test = int(num_machines * test_split)
        num_training = num_machines - num_test

        generator = torch.Generator().manual_seed(42)

        indices = torch.randperm(
            num_machines,
            generator=generator
        )

        train_indices: Tensor = indices[:num_training]
        validation_indices: Tensor = indices[num_training:]

        return train_indices, validation_indices

    @staticmethod
    def _fusion_validationdata(machines: list[MachineDataTensor]) -> MachineDataTensor:

        _geomtry = torch.cat(
            [machines[i].geometry_tensor for i in range(len(machines))],
            dim=0,
        )
        _maps = torch.cat(
            [machines[i].map_tensor for i in range(len(machines))],
            dim=0,
        )
        machinedata = MachineDataTensor(
            geometry_tensor=_geomtry,
            map_tensor=_maps,
        )
        return machinedata
