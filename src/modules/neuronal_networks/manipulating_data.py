
import math
import numpy as np
import torch
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

from config.config_mcad_parameters import *
from modules.data_managment.csv_io import import_data_from_csv
from config.config_paths import *


class DataTensor:

    def __init__(
            self,
            filename: str,
            label_axis: tuple[str, str, str] = OP_NEURONS,
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
        print(f"Found {self.num_machines} machines")

        self.machine_geometry_tensor = self._create_machine_geometry_tensor()
        self.operation_point_tensor = self._create_operation_point_tensor()
        self.map_tensor: torch.Tensor = self._create_map_tensor()

        (
            self.geo_train,
            self.geo_validation,
            self.map_train,
            self.map_validation,
            self.op_train,
            self.op_validation,
        ) = self._split_data(test_split=0.2)

        (
            self.geo_train_norm,
            self.geo_validation_norm,
            self.geo_mean,
            self.geo_std,
        ) = self._normalize_data(
            self.geo_train,
            self.geo_validation
        )

        (
            self.map_train_norm,
            self.map_validation_norm,
            self.map_mean,
            self.map_std,
        ) = self._normalize_data(
            self.map_train,
            self.map_validation
        )

        (
            self.op_train_norm,
            self.op_validation_norm,
            self.op_mean,
            self.op_std,
        ) = self._normalize_data(
            self.op_train,
            self.op_validation
        )


    @staticmethod
    def _get_data(filename: str) -> pd.DataFrame:
        data_df = import_data_from_csv(PATH_DATA_MCAD_RESULTS / filename)
        return data_df


    def _recalculate_data(self) -> None:
        conductor_area = (
                self.data_df["PeakCurrent"]
                / (np.sqrt(2) * self.data_df["RMSCurrentDensity"])
        )
        # Auf das nächste Vielfache von 0,05 runden.
        self.data_df[OP_NEURONS[1]] = (
            self.data_df["RMSPhaseCurrent_D"] / conductor_area * 20
        ).round() / 20
        self.data_df[OP_NEURONS[2]] = (
            self.data_df["RMSPhaseCurrent_Q"] / conductor_area * 20
        ).round() / 20

        d_si = self.data_df["Stator_Bore"]
        d_sa = self.data_df["Stator_Lam_Dia"]
        radial_width = (d_sa - d_si) / 2

        self.data_df[LAMBDA_LIST[0]] = d_si / d_sa
        self.data_df[LAMBDA_LIST[1]] = self.data_df["Magnet_Arc_[ED]"] / 180
        self.data_df[LAMBDA_LIST[2]] = self.data_df["Airgap"] / self.data_df["Magnet_Length"]

        # b_S / tau_s, b_S / d_S, h_PM / ((D_Sa - D_Si) / 2)
        self.data_df[LAMBDA_LIST[3]] = (
            self.data_df["Slot_Number"] * self.data_df["Slot_Width"] / (math.pi * d_si)
        )
        self.data_df[LAMBDA_LIST[4]] = self.data_df["Slot_Width"] / self.data_df["Slot_Depth"]
        self.data_df[LAMBDA_LIST[5]] = self.data_df["Magnet_Thickness"] / radial_width

        # d_emb / (radial_width - h_PM)
        self.data_df[LAMBDA_LIST[6]] = (
            self.data_df["Magnet_Embed_Depth"]
            / (radial_width - self.data_df["Magnet_Thickness"])
        )
        # (l_S - d_S) / tau_p, with tau_p = pi * (D_Sa + D_Si) / (2 * p)
        self.data_df[LAMBDA_LIST[7]] = (
            2 * self.data_df["Pole_Number"]
            * (self.data_df["Stator_Lam_Length"] - self.data_df["Slot_Depth"])
            / (math.pi * (d_sa + d_si))
        )
        self.data_df[LAMBDA_LIST[8]] = self.data_df["Magnet_Length"] / self.data_df["Rotor_Lam_Length"]



    def _clean_data(self) -> None:

        _machines = self._identify_machines_in_csv()
        points_per_map = math.prod(self.map_shape)
        _indices_to_delete = []

        maps_with_duplicate_points = 0
        uncomplete_maps = 0

        for idx_machine, (geometry_tuple, map_df) in enumerate(_machines):

            operating_points = map_df[self.label_axis]

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
            MACHINE_DEFINITION,
            sort=False
        )

        return _machines


    def _identify_map_axis_values(self)-> list[list]:

        _axis = []
        for axe_label in self.label_axis:
            _axis.append(sorted(self.data_df[axe_label].unique()))
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

        _machine_geometry = np.full(
            (
                self.num_machines,
                *self.map_shape,
                len(MACHINE_DEFINITION),
            ),
            np.nan,
            dtype=np.float32
        )

        # go through machines
        for idx_machine, (geometry_tuple, map_df) in enumerate(self.machines):
            # go through each op
            for _, row in map_df.iterrows():
                map_indices = [
                    self.axis_indices[axe][row[label]] for axe, label in enumerate(self.label_axis)
                ]  # list with three indices: index on the axe "axe" of the value row[label]
                _machine_geometry[
                    idx_machine,
                    *map_indices
                ] = geometry_tuple

        _geometry_tensor = torch.tensor(_machine_geometry, dtype=torch.float32)

        return _geometry_tensor


    def _create_operation_point_tensor(self) -> torch.Tensor:

        _operation_points_array = np.full(
            (
                self.num_machines,
                *self.map_shape,
                len(self.map_shape)
            ),
            np.nan,
            dtype=np.float32
        )

        # go through machines
        for idx_machine, (geometry_tuple, map_df) in enumerate(self.machines):
            # go through each op
            for _, row in map_df.iterrows():
                map_indices = [
                    self.axis_indices[axe][row[label]] for axe, label in enumerate(self.label_axis)
                ]  # list with three indices: index on the axe "axe" of the value row[label]

                for axe, label in enumerate(self.label_axis):
                    _operation_points_array[
                        idx_machine,
                        *map_indices,
                        axe
                    ] = row[label]

        _op_tensor = torch.tensor(_operation_points_array, dtype=torch.float32)

        return _op_tensor


    def _create_map_tensor(self) -> torch.Tensor:

        _map_array = np.full(
            (
                self.num_machines,
                *self.map_shape,
                len(OUTPUT_NEURON_LIST)
            ),
            np.nan,
            dtype=np.float32
        )

        #go through machines
        for idx_machine, (geometry_tuple, map_df) in enumerate(self.machines):
            #go through each op
            for _, row in map_df.iterrows():
                map_indices = [
                    self.axis_indices[axe][row[label]] for axe, label in enumerate(self.label_axis)
                ] #list with three indices: index on the axe "axe" of the value row[label]

                #go through each channel for op
                for idx_channel, channel in enumerate(OUTPUT_NEURON_LIST):
                    _map_array[
                        idx_machine,
                        *map_indices,
                        idx_channel
                    ] = row[channel]

        _map_tensor = torch.tensor(_map_array, dtype=torch.float32)

        nan_indices = torch.isnan(_map_tensor).nonzero(as_tuple=False)
        num_nan = nan_indices.size(0)
        if num_nan > 0:
            raise ValueError(
                f"Map tensor contains {num_nan} NaN values\n"
                f"Index order (zero-based): (machine, channel, {', '.join(self.label_axis)})\n"
                f"NaN indices: {nan_indices.tolist()}"
            )
        return _map_tensor


    def _split_data(
            self,
            test_split: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

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
        op_train = self.operation_point_tensor[train_indices]
        op_validation = self.operation_point_tensor[validation_indices]

        return x_train, x_validation, y_train, y_validation, op_train, op_validation


    @staticmethod
    def _normalize_data(
            train_tensor: torch.Tensor,
            validation_tansor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        dim_ = tuple(range(train_tensor.ndim - 1))

        mean = train_tensor.mean(dim=dim_, keepdim=True)
        std = torch.clamp(
            train_tensor.std(
                dim=dim_,
                keepdim=True,
                correction=0
            ),
            min=1e-8
        )
        normalized_train_tensor = (train_tensor - mean) / std
        normalized_validation_tensor = (validation_tansor - mean) / std

        return normalized_train_tensor, normalized_validation_tensor, mean, std
