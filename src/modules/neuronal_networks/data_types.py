from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


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

    def select_channel(self, channel_index: int) -> MachineDataTensor:
        if self.map_tensor.ndim != 5:
            raise ValueError("Expected a five-dimensional map tensor.")

        if (
            self.geometry_tensor.ndim != 2
            or self.geometry_tensor.shape[0] != self.map_tensor.shape[0]
        ):
            raise ValueError(
                "Geometry and map tensors must describe the same machines."
            )

        if (
            type(channel_index) is not int
            or not 0 <= channel_index < self.map_tensor.shape[-1]
        ):
            raise ValueError("Output channel index is out of range.")

        return MachineDataTensor(
            geometry_tensor=self.geometry_tensor.detach().clone(),
            map_tensor=self.map_tensor[..., channel_index : channel_index + 1]
            .detach()
            .clone(),
        )


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
