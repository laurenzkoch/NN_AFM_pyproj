import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pandas.core.groupby.generic import DataFrameGroupBy
from torch import Tensor

from src.config.config_mcad_parameters import (
    AXIS_LIST,
    CSV_PARAMETERS,
    D_CURRENT_DENSITY,
    GEO_PARAMETERS,
    GEO_PARAMETERS_MIN_MAX_DICT,
    LAMBDA_LIST,
    MACHINE_DEFINITION,
    MAP_SIZE,
    OUTPUT_NEURON_LIST,
    Q_CURRENT_DENSITY,
    SHAFT_SPEED,
)
from src.modules.data_managment.csv_io import import_data_from_csv
from src.modules.data_managment.experiment_utils import (
    file_sha256,
    geometry_ids,
    require_disjoint_split_ids,
)

logger = logging.getLogger(__name__)


@dataclass
class MachineDataTensor:
    geometry_tensor: Tensor
    map_tensor: Tensor

    def shuffle(self) -> None:
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
    map_axes: dict[str, np.ndarray]
    target_minimum: Tensor
    target_denominator: Tensor
    split_machine_ids: dict[str, tuple[str, ...]]
    source_hashes: dict[str, str]


class DataLoader:
    def __init__(self, training_directory: Path) -> None:
        self.training_directory = training_directory

        self.map_axes = {
            AXIS_LIST[0]: np.linspace(
                *SHAFT_SPEED,
                MAP_SIZE[0],
            ),
            AXIS_LIST[1]: np.round(np.linspace(*D_CURRENT_DENSITY, MAP_SIZE[1]) * 20)
            / 20,
            AXIS_LIST[2]: np.round(np.linspace(*Q_CURRENT_DENSITY, MAP_SIZE[2]) * 20)
            / 20,
        }

        self.map_indices = {
            name: {value: index for index, value in enumerate(values)}
            for name, values in self.map_axes.items()
        }

        self.expected_operating_points = pd.MultiIndex.from_product(
            [self.map_axes[name] for name in AXIS_LIST],
            names=AXIS_LIST,
        )

    def load(
        self, filename: str, test_filenames: tuple[str, ...] = ()
    ) -> TrainValidationTestDataTensors:
        names = (filename, *test_filenames)

        if len(set(names)) != len(names):
            raise ValueError("Each CSV source must be listed once")

        records = []
        source_hashes = {}
        for name in names:
            if Path(name).name != name or Path(name).suffix.lower() != ".csv":
                raise ValueError("Use CSV filenames without a directory")
            _file_path = self.training_directory / name
            source_hashes[name] = file_sha256(_file_path)
            _dataframe: pd.DataFrame = import_data_from_csv(_file_path)
            self._test_columns(_dataframe)
            _dataframe = _dataframe[CSV_PARAMETERS].copy()
            _dataframe = self._recalculate_data(_dataframe)
            _dataframe = self._clean_data(_dataframe)
            if _dataframe.empty:
                raise ValueError(f"No complete reference-grid maps: {name}")
            _machines_groups = self._identify_machines_in_csv(_dataframe)
            _machines = self._create_tensors(
                machine_groups=_machines_groups,
                map_indices=self.map_indices,
                num_machines=len(_machines_groups),
            )
            ids = geometry_ids(_machines.geometry_tensor)
            _machines.geometry_tensor = self._normalize_geo_tensor(
                _machines.geometry_tensor
            )
            records.append((_machines, ids))

        main, main_ids = records[0]
        train_idx, val_idx = self._split_indices(
            main.geometry_tensor.shape[0], test_split=0.2
        )
        train = MachineDataTensor(
            main.geometry_tensor[train_idx], main.map_tensor[train_idx]
        )
        validation = MachineDataTensor(
            main.geometry_tensor[val_idx], main.map_tensor[val_idx]
        )
        test = (
            self._fusion_validationdata([record[0] for record in records[1:]])
            if len(records) > 1
            else None
        )

        split_ids = {
            "train": tuple(main_ids[int(i)] for i in train_idx),
            "validation": tuple(main_ids[int(i)] for i in val_idx),
            "test": tuple(key for _, ids in records[1:] for key in ids),
        }
        require_disjoint_split_ids(split_ids)
        if len(split_ids["test"]) != len(set(split_ids["test"])):
            raise ValueError("Duplicate machines between external test sources")

        minimum, maximum = self._calc_min_max(train.map_tensor)
        denominator = (maximum - minimum).clamp_min(1e-8)
        for split in (train, validation, test):
            if split is not None:
                split.map_tensor = (split.map_tensor - minimum) / denominator
        return TrainValidationTestDataTensors(
            train=train,
            validation=validation,
            test=test,
            map_axes={name: axis.copy() for name, axis in self.map_axes.items()},
            target_minimum=minimum.detach().cpu().clone(),
            target_denominator=denominator.detach().cpu().clone(),
            split_machine_ids=split_ids,
            source_hashes=source_hashes,
        )

    @staticmethod
    def check_split_overlap(
        train_geometry: torch.Tensor,
        validation_geometry: torch.Tensor,
        test_geometry: torch.Tensor,
    ) -> None:

        def machine_keys(
            geometry: Tensor,
        ) -> set[tuple[float, ...]]:
            return {
                tuple(float(value) for value in row)
                for row in geometry.detach().cpu().tolist()
            }

        train_keys = machine_keys(train_geometry)
        validation_keys = machine_keys(validation_geometry)
        test_keys = machine_keys(test_geometry)

        if train_keys & validation_keys:
            raise ValueError("Train and validation contain the same machines")

        if test_keys & (train_keys | validation_keys):
            raise ValueError(
                "Test contains machines already used for training/selection"
            )

    @staticmethod
    def _test_columns(df: pd.DataFrame) -> None:
        missing: list[str] = [
            column for column in CSV_PARAMETERS if column not in df.columns
        ]

        if missing:
            logger.error("Missing CSV columns: %s", missing)
            raise ValueError(f"Missing CSV columns: {missing}")

    @staticmethod
    def _recalculate_data(df: pd.DataFrame) -> pd.DataFrame:
        raw_values: NDArray[np.float64] = df[CSV_PARAMETERS].to_numpy(dtype=np.float64)

        if not bool(np.isfinite(raw_values).all()):
            raise ValueError("Source data contains NaN or infinite values.")

        positive_columns: tuple[str, ...] = (
            "PeakCurrent",
            "RMSCurrentDensity",
            "Stator_Bore",
            "Stator_Lam_Dia",
            "Magnet_Length",
            "Slot_Depth",
            "Rotor_Lam_Length",
        )

        for column in positive_columns:
            if bool((df[column] <= 0).any()):
                raise ValueError(f"'{column}' must be strictly positive.")

        radial_width_values: NDArray[np.float64] = (
            (df["Stator_Lam_Dia"] - df["Stator_Bore"]) / 2
        ).to_numpy(dtype=np.float64)

        remaining_width: NDArray[np.float64] = radial_width_values - df[
            "Magnet_Thickness"
        ].to_numpy(dtype=np.float64)

        if bool((radial_width_values <= 0).any()):
            raise ValueError("Radial width must be strictly positive.")

        if bool((remaining_width <= 0).any()):
            raise ValueError("Radial width minus magnet thickness must be positive.")

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
        df[LAMBDA_LIST[6]] = df["Magnet_Embed_Depth"] / (
            radial_width - df["Magnet_Thickness"]
        )
        # (l_S - d_S) / tau_p, with tau_p = pi * (D_Sa + D_Si) / (2 * p)
        df[LAMBDA_LIST[7]] = (
            2
            * df["Pole_Number"]
            * (df["Stator_Lam_Length"] - df["Slot_Depth"])
            / (math.pi * (d_sa + d_si))
        )
        df[LAMBDA_LIST[8]] = df["Magnet_Length"] / df["Rotor_Lam_Length"]

        return df

    @staticmethod
    def _identify_machines_in_csv(df: pd.DataFrame) -> DataFrameGroupBy:
        _machines = df.groupby(MACHINE_DEFINITION, sort=False)
        return _machines

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        remove = []
        for geometry, frame in self._identify_machines_in_csv(df):
            points = frame[AXIS_LIST]
            if points.duplicated().any():
                remove.extend(frame.index)
                logger.warning("Duplicate operating points: %s", geometry)
                continue
            actual = pd.MultiIndex.from_frame(points)
            missing = self.expected_operating_points.difference(actual)
            extra = actual.difference(self.expected_operating_points)
            if len(missing) or len(extra):
                remove.extend(frame.index)
                logger.warning(
                    "Wrong map grid: %s, missing=%s, extra=%s",
                    geometry,
                    len(missing),
                    len(extra),
                )
        return df.drop(remove)

    @staticmethod
    def _create_tensors(
        machine_groups: DataFrameGroupBy,
        map_indices: dict[str, dict[float, int]],
        num_machines: int,
    ) -> MachineDataTensor:
        def _get_map_coordinates(
            _row: pd.Series, idx_dict: dict[str, dict[float, int]]
        ) -> tuple[int, ...]:
            coordinates = tuple[int, ...](
                idx_dict[axe_label][_row[axe_label]] for axe_label in AXIS_LIST
            )
            return coordinates

        _machine_geometry_array = np.full(
            (
                num_machines,
                len(MACHINE_DEFINITION),
            ),
            np.nan,
            dtype=np.float32,
        )
        _map_array = np.full(
            (num_machines, *MAP_SIZE, len(OUTPUT_NEURON_LIST)), np.nan, dtype=np.float32
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
                "Index order (zero-based): "
                f"(machine,  {', '.join(AXIS_LIST)}, channel)\n"
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

        indices = torch.randperm(num_machines, generator=generator)

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
