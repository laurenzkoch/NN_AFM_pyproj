import math

import torch.nn as nn
from torch import Tensor

from src.config.config_mcad_parameters import MAP_SIZE


class Conv3dAutoencoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        features: tuple[int, ...],
        input_channels: int = 1,
        map_shape: tuple[int, ...] = MAP_SIZE,
        kernel_size: int = 3,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        if type(latent_dim) is not int or latent_dim < 1:
            raise ValueError("latent_dim must be a positive integer.")

        if type(input_channels) is not int or input_channels < 1:
            raise ValueError("input_channels must be a positive integer.")

        if not features or any(
            type(value) is not int or value < 1 for value in features
        ):
            raise ValueError("features must contain positive hidden feature counts.")

        if any(
            current <= previous for previous, current in zip(features, features[1:])
        ):
            raise ValueError("Hidden feature counts must be increasing.")

        if len(map_shape) != 3 or any(
            type(value) is not int or value < 1 for value in map_shape
        ):
            raise ValueError("map_shape must contain three positive integers.")

        if type(kernel_size) is not int or kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")

        self.model_state_dict: dict[str, Tensor]

        self.input_channels: int = input_channels
        self.latent_dim: int = latent_dim
        self.features: tuple[int, ...] = features
        self.map_shape: tuple[int, ...] = map_shape
        self.kernel_size: int = kernel_size
        self.training_duration_seconds: float | None = None

        padding: int = kernel_size // 2

        encoder_layers: list[nn.Module] = []
        in_channels: int = input_channels
        for out_channels in features:
            encoder_layers.append(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
            encoder_layers.append(activation())
            in_channels = out_channels

        decoder_layers: list[nn.Module] = []
        decoder_channels: tuple[int, ...] = (
            *reversed(features[:-1]),
            input_channels,
        )
        in_channels = features[-1]
        for idx, out_channels in enumerate(decoder_channels):
            decoder_layers.append(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
            if idx < len(decoder_channels) - 1:
                decoder_layers.append(activation())
            in_channels = out_channels

        self.encoder_conv = nn.Sequential(*encoder_layers)
        self.encoder_linear = nn.Linear(features[-1] * math.prod(map_shape), latent_dim)

        self.decoder_conv = nn.Sequential(*decoder_layers)
        self.decoder_linear = nn.Linear(latent_dim, features[-1] * math.prod(map_shape))

    def encode(self, x: Tensor) -> Tensor:
        x = x.permute(0, 4, 1, 2, 3)

        x = self.encoder_conv(x)
        # before: (batch, *map_shape, num_output_neurons)
        # after: (batch, *map_shape, features[-1],)

        x = x.flatten(start_dim=1)
        # before: (batch, *map_shape, features[-1])
        # after: (batch, features[-1] * math.prod(map_shape))

        z = self.encoder_linear(x)
        # before:    (batch, features[-1] * math.prod(map_shape))
        # after:     (batch, latent_dim)

        return z

    def decode(self, z: Tensor) -> Tensor:
        x = self.decoder_linear(z)
        # before: (batch, latent_dim)
        # after: (batch, features[-1] * math.prod(map_shape))

        x = x.view(-1, self.features[-1], *self.map_shape)
        # before: (batch, features[-1] * math.prod(map_shape))
        # after: (batch, *map_shape, features[-1])

        x = self.decoder_conv(x)
        # before: (batch, *map_shape,features[-1])
        # after: (batch, *map_shape, num_output_neurons)

        x = x.permute(0, 2, 3, 4, 1)

        return x

    def forward(self, x: Tensor) -> Tensor:
        z = self.encode(x)

        x_hat = self.decode(z)

        return x_hat
