from dataclasses import dataclass, replace

import torch
from torch import Tensor

from src.config.config_mcad_parameters import AXIS_LIST, OUTPUT_NEURON_LIST
from src.modules.neuronal_networks.manipulating_data import (
    MachineDataTensor,
    TrainValidationTestDataTensors,
)
from src.modules.neuronal_networks.NN_types.cnn_autoencoder import (
    Conv3dAutoencoder,
)


@dataclass
class AutoencoderCollection:
    autoencoders: tuple[Conv3dAutoencoder, ...]
    map_axes: dict[str, tuple[float, ...]]
    target_minimum: Tensor
    target_denominator: Tensor
    split_machine_ids: dict[str, tuple[str, ...]]
    source_hashes: dict[str, str]

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Return output channels in the globally configured order."""
        return tuple(OUTPUT_NEURON_LIST)

    def __post_init__(self) -> None:
        channel_count: int = len(self.channel_names)

        if channel_count == 0:
            raise ValueError("At least one output channel is required.")

        if len(set(self.channel_names)) != channel_count:
            raise ValueError("Output channel names must be unique.")

        if len(self.autoencoders) != channel_count:
            raise ValueError("Exactly one autoencoder per output channel is required.")

        if len({id(model) for model in self.autoencoders}) != channel_count:
            raise ValueError("Each output channel requires an independent autoencoder.")

        for name, values in (
            ("target_minimum", self.target_minimum),
            ("target_denominator", self.target_denominator),
        ):
            if values.shape != (channel_count,):
                raise ValueError(f"{name} must contain one value per output channel.")

            if not values.is_floating_point():
                raise ValueError(f"{name} must contain floating-point values.")

            if not bool(torch.isfinite(values).all().item()):
                raise ValueError(f"{name} contains non-finite values.")

        if not bool((self.target_denominator > 0).all().item()):
            raise ValueError("target_denominator must be strictly positive.")

        for autoencoder in self.autoencoders:
            if autoencoder.input_channels != 1:
                raise ValueError(
                    "Each autoencoder must process exactly one output channel."
                )

    def for_channel(self, channel_name: str) -> Conv3dAutoencoder:
        try:
            channel_index: int = self.channel_names.index(channel_name)
        except ValueError as exc:
            raise ValueError(
                f"No autoencoder available for channel: {channel_name}"
            ) from exc

        return self.autoencoders[channel_index]

    def prepare_data(
        self,
        data: TrainValidationTestDataTensors,
    ) -> TrainValidationTestDataTensors:

        channel_count: int = len(self.channel_names)

        old_minimum: Tensor = data.target_minimum.detach().cpu().double()
        old_denominator: Tensor = data.target_denominator.detach().cpu().double()
        new_minimum: Tensor = self.target_minimum.detach().cpu().double()
        new_denominator: Tensor = self.target_denominator.detach().cpu().double()

        for name, values in (
            ("old minimum", old_minimum),
            ("old denominator", old_denominator),
            ("CNN minimum", new_minimum),
            ("CNN denominator", new_denominator),
        ):
            if values.shape != (channel_count,):
                raise ValueError(f"Wrong number of normalization values: {name}")

            if not bool(torch.isfinite(values).all().item()):
                raise ValueError(f"Non-finite normalization values: {name}")

        if not (
            bool((old_denominator > 0).all().item())
            and bool((new_denominator > 0).all().item())
        ):
            raise ValueError("Normalization denominators must be strictly positive.")

        def rebase_split(
            split: MachineDataTensor,
        ) -> MachineDataTensor:
            source: Tensor = split.map_tensor

            if (
                source.ndim != 5
                or source.shape[-1] != channel_count
                or not source.is_floating_point()
            ):
                raise ValueError("Invalid map tensor for normalization conversion.")

            physical: Tensor = (
                source.detach().cpu().double() * old_denominator + old_minimum
            )
            normalized: Tensor = ((physical - new_minimum) / new_denominator).to(
                dtype=source.dtype,
                device=source.device,
            )

            if not bool(torch.isfinite(normalized).all().item()):
                raise FloatingPointError(
                    "Normalization conversion produced non-finite values."
                )

            return MachineDataTensor(
                geometry_tensor=split.geometry_tensor.detach().clone(),
                map_tensor=normalized,
            )

        prepared: TrainValidationTestDataTensors = replace(
            data,
            train=rebase_split(data.train),
            validation=rebase_split(data.validation),
            test=(rebase_split(data.test) if data.test is not None else None),
            target_minimum=self.target_minimum.detach().cpu().clone(),
            target_denominator=(self.target_denominator.detach().cpu().clone()),
        )

        self.validate_compatible(prepared)
        return prepared

    def validate_compatible(
        self,
        data: TrainValidationTestDataTensors,
    ) -> None:

        if set(self.map_axes) != set(AXIS_LIST) or set(data.map_axes) != set(AXIS_LIST):
            raise ValueError("Autoencoder map axes are missing or invalid.")

        for axis_name in AXIS_LIST:
            saved_axis: Tensor = torch.tensor(
                self.map_axes[axis_name],
                dtype=torch.float64,
            )
            current_axis: Tensor = torch.as_tensor(
                data.map_axes[axis_name],
                dtype=torch.float64,
            ).cpu()

            if not torch.equal(saved_axis, current_axis):
                raise ValueError(f"Autoencoder map axis differs: {axis_name}")

        expected_shape: tuple[int, ...] = tuple(
            len(self.map_axes[name]) for name in AXIS_LIST
        )

        for channel_name, autoencoder in zip(
            self.channel_names,
            self.autoencoders,
            strict=True,
        ):
            if tuple(autoencoder.map_shape) != expected_shape:
                raise ValueError(f"Autoencoder map shape differs: {channel_name}")

        for name, saved, current in (
            (
                "target_minimum",
                self.target_minimum,
                data.target_minimum,
            ),
            (
                "target_denominator",
                self.target_denominator,
                data.target_denominator,
            ),
        ):
            if not torch.equal(
                saved.detach().cpu().double(),
                current.detach().cpu().double(),
            ):
                raise ValueError(f"Autoencoder normalization differs: {name}")

        for split_name in ("train", "validation"):
            if (
                split_name not in self.split_machine_ids
                or split_name not in data.split_machine_ids
            ):
                raise ValueError(f"Missing dataset split: {split_name}")

            saved_ids: tuple[str, ...] = self.split_machine_ids[split_name]
            current_ids: tuple[str, ...] = data.split_machine_ids[split_name]

            if len(set(saved_ids)) != len(saved_ids) or len(set(current_ids)) != len(
                current_ids
            ):
                raise ValueError(f"Duplicate machine IDs in split: {split_name}")

        training_ids: set[str] = set(self.split_machine_ids["train"])
        validation_ids: set[str] = set(self.split_machine_ids["validation"])
        test_ids: set[str] = set(data.split_machine_ids.get("test", ()))

        if training_ids & validation_ids:
            raise ValueError("Autoencoder training and validation machines overlap.")

        if (training_ids | validation_ids) & test_ids:
            raise ValueError(
                "Test machines were used during autoencoder training "
                "or model selection."
            )

        for source_name in self.source_hashes.keys() & data.source_hashes.keys():
            if self.source_hashes[source_name] != data.source_hashes[source_name]:
                raise ValueError(f"Autoencoder source file has changed: {source_name}")

    @torch.no_grad()
    def reconstruct(self, map_tensor: Tensor) -> Tensor:
        """Reconstruct all output channels using their trained CNNs."""
        if (
            map_tensor.ndim != 5
            or map_tensor.shape[-1] != len(self.channel_names)
            or map_tensor.shape[0] == 0
            or not map_tensor.is_floating_point()
        ):
            raise ValueError(
                "Expected a nonempty map tensor with one channel per configured output."
            )

        if not bool(torch.isfinite(map_tensor).all().item()):
            raise ValueError("Input maps contain non-finite values.")

        map_shape: tuple[int, ...] = tuple(map_tensor.shape[1:-1])

        for channel_name, autoencoder in zip(
            self.channel_names,
            self.autoencoders,
            strict=True,
        ):
            if tuple(autoencoder.map_shape) != map_shape:
                raise ValueError(
                    f"Input map dimensions differ from the CNN: {channel_name}"
                )

        reconstructions: list[Tensor] = []

        for channel_index, autoencoder in enumerate(self.autoencoders):
            autoencoder.eval()

            channel_map: Tensor = map_tensor[..., channel_index : channel_index + 1]
            reconstruction: Tensor = autoencoder(channel_map)

            if reconstruction.shape != channel_map.shape:
                raise ValueError("CNN reconstruction dimensions differ from the input.")

            if not bool(torch.isfinite(reconstruction).all().item()):
                raise FloatingPointError(
                    "CNN reconstruction contains non-finite values."
                )

            reconstructions.append(reconstruction)

        return torch.cat(reconstructions, dim=-1)

    def freeze(self) -> None:
        """Use evaluation mode and disable parameter gradient calculation."""
        for autoencoder in self.autoencoders:
            autoencoder.eval()
            autoencoder.requires_grad_(False)

            for parameter in autoencoder.parameters():
                parameter.grad = None
