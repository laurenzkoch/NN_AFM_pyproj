import math
from typing import Literal, TypedDict

import torch
from torch import Tensor, nn

from src.modules.neuronal_networks.NN_types.cnn_autoencoder import (
    Conv3dAutoencoder,
)
from src.modules.neuronal_networks.NN_types.mlp_with_torch import MLP


class MLPCheckpoint(TypedDict):
    network_type: Literal["MLP"]
    activation: Literal["ReLU"]
    num_input_neurons: int
    hidden_layers: tuple[int, ...]
    output_dims: tuple[int, ...]
    training_duration_seconds: float | None
    state_dict: dict[str, Tensor]


class AutoencoderCheckpoint(TypedDict):
    network_type: Literal["CNN"]
    activation: Literal["ReLU"]
    input_channels: int
    features: tuple[int, ...]
    latent_dim: int
    map_shape: tuple[int, ...]
    kernel_size: int
    training_duration_seconds: float | None
    state_dict: dict[str, Tensor]


def checked_training_duration(value: object) -> float | None:
    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Training duration must be numeric or None.")

    duration: float = float(value)

    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Training duration must be finite and nonnegative.")

    return duration


def _copy_state(network: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in network.state_dict().items()
    }


def pack_mlp(model: MLP) -> MLPCheckpoint:
    allowed_types: tuple[type[nn.Module], ...] = (
        nn.Sequential,
        nn.Linear,
        nn.ReLU,
    )

    for module in tuple(model.modules())[1:]:
        if type(module) not in allowed_types:
            raise ValueError(f"Unsupported MLP module: {type(module).__name__}")

    return {
        "network_type": "MLP",
        "activation": "ReLU",
        "num_input_neurons": model.num_input_neurons,
        "hidden_layers": tuple(model.hidden_layer_neurons),
        "output_dims": tuple(model.output_dims),
        "training_duration_seconds": checked_training_duration(
            model.training_duration_seconds
        ),
        "state_dict": _copy_state(model),
    }


def pack_autoencoder(
    model: Conv3dAutoencoder,
) -> AutoencoderCheckpoint:

    allowed_types: tuple[type[nn.Module], ...] = (
        nn.Sequential,
        nn.Conv3d,
        nn.Linear,
        nn.ReLU,
    )

    for module in tuple(model.modules())[1:]:
        if type(module) not in allowed_types:
            raise ValueError(f"Unsupported CNN module: {type(module).__name__}")

    return {
        "network_type": "CNN",
        "activation": "ReLU",
        "input_channels": model.input_channels,
        "features": tuple(model.features),
        "latent_dim": model.latent_dim,
        "map_shape": tuple(model.map_shape),
        "kernel_size": model.kernel_size,
        "training_duration_seconds": checked_training_duration(
            model.training_duration_seconds
        ),
        "state_dict": _copy_state(model),
    }


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")

    return value


def _integer_tuple(
    value: object,
    name: str,
    *,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must contain a list or tuple.")

    if not value and not allow_empty:
        raise ValueError(f"{name} must not be empty.")

    return tuple(_positive_integer(item, name) for item in value)


def _restore_state(
    model: nn.Module,
    value: object,
) -> None:

    if not isinstance(value, dict):
        raise TypeError("state_dict must contain a dictionary.")

    state: dict[str, Tensor] = {}

    for name, tensor in value.items():
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            raise TypeError("state_dict must map parameter names to tensors.")

        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"Non-finite values in model state: {name}")

        state[name] = tensor.detach().cpu()

    model.load_state_dict(state, strict=True)
    model.eval()


def unpack_mlp(checkpoint: object) -> MLP:

    if not isinstance(checkpoint, dict):
        raise TypeError("An MLP checkpoint must contain a dictionary.")

    if checkpoint.get("network_type") != "MLP":
        raise ValueError("Expected an MLP checkpoint.")

    if checkpoint.get("activation") != "ReLU":
        raise ValueError("Unsupported MLP activation.")

    model: MLP = MLP(
        num_input_neurons=_positive_integer(
            checkpoint.get("num_input_neurons"),
            "num_input_neurons",
        ),
        hidden_layer_neurons=_integer_tuple(
            checkpoint.get("hidden_layers"),
            "hidden_layers",
            allow_empty=True,
        ),
        output_dims=_integer_tuple(
            checkpoint.get("output_dims"),
            "output_dims",
        ),
    )

    _restore_state(model, checkpoint.get("state_dict"))

    model.training_duration_seconds = checked_training_duration(
        checkpoint.get("training_duration_seconds")
    )

    return model


def unpack_autoencoder(checkpoint: object) -> Conv3dAutoencoder:

    if not isinstance(checkpoint, dict):
        raise TypeError("A CNN checkpoint must contain a dictionary.")

    if checkpoint.get("network_type") != "CNN":
        raise ValueError("Expected a CNN checkpoint.")

    if checkpoint.get("activation") != "ReLU":
        raise ValueError("Unsupported CNN activation.")

    model: Conv3dAutoencoder = Conv3dAutoencoder(
        input_channels=_positive_integer(
            checkpoint.get("input_channels"),
            "input_channels",
        ),
        features=_integer_tuple(
            checkpoint.get("features"),
            "features",
        ),
        latent_dim=_positive_integer(
            checkpoint.get("latent_dim"),
            "latent_dim",
        ),
        map_shape=_integer_tuple(
            checkpoint.get("map_shape"),
            "map_shape",
        ),
        kernel_size=_positive_integer(
            checkpoint.get("kernel_size"),
            "kernel_size",
        ),
    )

    _restore_state(model, checkpoint.get("state_dict"))

    model.training_duration_seconds = checked_training_duration(
        checkpoint.get("training_duration_seconds")
    )

    return model
