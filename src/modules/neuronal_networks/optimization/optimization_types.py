from dataclasses import dataclass
from typing import Literal

type NetworkType = Literal["MLP", "CNN"]


def checked_network_type(value: object) -> NetworkType:
    if value == "MLP":
        return "MLP"
    if value == "CNN":
        return "CNN"

    raise ValueError("network_type must be 'MLP' or 'CNN'.")


@dataclass(frozen=True)
class NetworkSearchSpace:
    max_num_layers: int
    max_num_units_per_layer: int
    unit_step: int
    min_latent_dim: int = 1
    max_latent_dim: int = 32
    latent_step: int = 1

    def __post_init__(self) -> None:
        positive_settings: tuple[int, ...] = (
            self.max_num_layers,
            self.max_num_units_per_layer,
            self.unit_step,
            self.min_latent_dim,
            self.max_latent_dim,
            self.latent_step,
        )

        if any(type(value) is not int or value < 1 for value in positive_settings):
            raise ValueError("Search settings must be positive integers.")

        if self.unit_step > self.max_num_units_per_layer:
            raise ValueError("unit_step exceeds max_num_units_per_layer.")

        if self.min_latent_dim > self.max_latent_dim:
            raise ValueError("min_latent_dim exceeds max_latent_dim.")

        if (self.max_latent_dim - self.min_latent_dim) % self.latent_step != 0:
            raise ValueError(
                "The latent dimension range must be divisible by latent_step."
            )

    def unit_choices(self) -> tuple[int, ...]:
        """Return the candidate neuron or convolutional feature counts."""
        return tuple(
            range(
                self.unit_step,
                self.max_num_units_per_layer + 1,
                self.unit_step,
            )
        )


@dataclass(frozen=True)
class NetworkArchitecture:
    network_type: NetworkType
    layers: tuple[int, ...]
    latent_dim: int | None = None
    input_channels: int = 1

    def __post_init__(self) -> None:
        if self.network_type not in ("MLP", "CNN"):
            raise ValueError("network_type must be 'MLP' or 'CNN'.")

        if not self.layers or any(
            type(value) is not int or value < 1 for value in self.layers
        ):
            raise ValueError("Layer sizes must be positive integers.")

        if self.network_type == "MLP":
            if self.latent_dim is not None:
                raise ValueError(
                    "An MLP obtains its output size from the map "
                    "or the supplied trained autoencoder."
                )
            return

        if type(self.input_channels) is not int or self.input_channels < 1:
            raise ValueError("input_channels must be a positive integer.")

        if any(
            current <= previous
            for previous, current in zip(
                self.layers,
                self.layers[1:],
            )
        ):
            raise ValueError("CNN feature counts must be strictly increasing.")

        if type(self.latent_dim) is not int or self.latent_dim < 1:
            raise ValueError("A CNN requires a positive latent dimension.")


@dataclass(frozen=True)
class ChannelOptimizationResult:
    channel_index: int
    architecture: NetworkArchitecture
    mse: float
    r_squared: float | None
    trial_number: int
