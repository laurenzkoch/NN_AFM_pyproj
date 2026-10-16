from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypedDict, cast

import torch
from torch import Tensor

from src.config.config_mcad_parameters import (
    AXIS_LIST,
    GEO_PARAMETERS_MIN_MAX_DICT,
    MACHINE_DEFINITION,
    OUTPUT_NEURON_LIST,
)
from src.modules.neuronal_networks.manipulating_data import (
    TrainValidationTestDataTensors,
)
from src.modules.neuronal_networks.NN_types.mlp_with_torch import MLP
from src.modules.neuronal_networks.surrogate_model import (
    SubSurrogateModel,
    SurrogateModel,
)

type JSONValue = (
    str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]
)


class SurrogateCheckpoint(TypedDict):
    format_version: Literal[1]
    model_type: Literal["direct_mlp_ensemble"]
    activation: Literal["ReLU"]
    feature_names: list[str]
    channel_names: list[str]
    geometry_bounds: dict[str, tuple[float, float]]
    map_axes: dict[str, list[float]]
    axis_names: list[str]
    output_dims: tuple[int, ...]
    hidden_layers: tuple[int, ...]
    target_minimum: Tensor
    target_denominator: Tensor
    split_machine_ids: dict[str, tuple[str, ...]]
    source_hashes: dict[str, str]
    seed: int
    metrics: dict[str, JSONValue]
    model_states: list[dict[str, Tensor]]


def save_bundle(
    path: Path,
    model: SurrogateModel,
    hidden_layers: tuple[int, ...],
    data: TrainValidationTestDataTensors,
    metrics: dict[str, JSONValue],
    seed: int,
) -> None:
    if any(part.cnn_autoencoder is not None for part in model.submodels):
        raise ValueError("This bundle format supports direct MLPs only")
    if len(model.submodels) != len(OUTPUT_NEURON_LIST):
        raise ValueError("Wrong number of channel models")
    for channel_index, part in enumerate(model.submodels):
        layers: list[torch.nn.Module] = list(part.mlp.network.children())

        for layer in layers:
            if type(layer) not in (torch.nn.Linear, torch.nn.ReLU):
                raise ValueError(
                    f"Channel {channel_index}: unsupported layer "
                    f"'{type(layer).__name__}'. "
                    "This bundle format supports Linear and ReLU only."
                )

    checkpoint: SurrogateCheckpoint = {
        "format_version": 1,
        "model_type": "direct_mlp_ensemble",
        "activation": "ReLU",
        "feature_names": list(MACHINE_DEFINITION),
        "channel_names": list(OUTPUT_NEURON_LIST),
        "geometry_bounds": dict(GEO_PARAMETERS_MIN_MAX_DICT),
        "map_axes": {k: v.tolist() for k, v in data.map_axes.items()},
        "axis_names": list(data.map_axes),
        "output_dims": tuple(data.train.map_tensor.shape[1:-1]) + (1,),
        "hidden_layers": tuple(hidden_layers),
        "target_minimum": data.target_minimum,
        "target_denominator": data.target_denominator,
        "split_machine_ids": data.split_machine_ids,
        "source_hashes": data.source_hashes,
        "seed": int(seed),
        "metrics": metrics,
        "model_states": [
            {k: v.detach().cpu().clone() for k, v in part.mlp.state_dict().items()}
            for part in model.submodels
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_bundle(
    path: Path,
) -> tuple[SurrogateModel, SurrogateCheckpoint]:
    raw: object = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(raw, dict):
        raise ValueError("The model bundle must contain a dictionary.")

    missing_fields: set[str] = set(SurrogateCheckpoint.__required_keys__) - set(raw)

    if missing_fields:
        raise ValueError(f"Missing checkpoint fields: {sorted(missing_fields)}")

    checkpoint: SurrogateCheckpoint = cast(SurrogateCheckpoint, raw)

    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported bundle version")
    if checkpoint.get("model_type") != "direct_mlp_ensemble":
        raise ValueError("Unsupported model type")
    if checkpoint.get("activation") != "ReLU":
        raise ValueError("Unsupported activation")
    if checkpoint.get("feature_names") != list(MACHINE_DEFINITION):
        raise ValueError(
            "Saved feature names or their order differ "
            "from the current project configuration."
        )
    if checkpoint.get("channel_names") != list(OUTPUT_NEURON_LIST):
        raise ValueError(
            "Saved output channel names or their order differ "
            "from the current project configuration."
        )
    for key in ("target_minimum", "target_denominator"):
        value: object = checkpoint.get(key)

        if not isinstance(value, Tensor):
            raise ValueError(f"'{key}' must be a tensor.")

        if value.shape != (len(OUTPUT_NEURON_LIST),):
            raise ValueError(f"'{key}' must contain one value per output channel.")

        if not value.is_floating_point():
            raise ValueError(f"'{key}' must contain floating-point values.")

        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"'{key}' contains non-finite values.")

        if key == "target_denominator":
            if not bool((value > 0).all().item()):
                raise ValueError("'target_denominator' must be strictly positive.")
    if checkpoint.get("axis_names") != list(AXIS_LIST):
        raise ValueError(
            "Saved axis names or their order differ "
            "from the current project configuration."
        )

    axes: object = checkpoint.get("map_axes")

    if not isinstance(axes, dict) or set(axes) != set(AXIS_LIST):
        raise ValueError("Saved map axes are missing or invalid.")

    axis_sizes: list[int] = []

    for name in AXIS_LIST:
        axis: object = axes[name]

        if not isinstance(axis, list) or not axis:
            raise ValueError(f"Axis '{name}' must be a non-empty list.")

        try:
            axis_values: Tensor = torch.tensor(
                axis,
                dtype=torch.float64,
            )
        except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
            raise ValueError(f"Axis '{name}' contains invalid coordinates.") from exc

        if axis_values.ndim != 1:
            raise ValueError(f"Axis '{name}' must be one-dimensional.")

        if not bool(torch.isfinite(axis_values).all().item()):
            raise ValueError(f"Axis '{name}' contains non-finite values.")

        if not bool((axis_values[1:] > axis_values[:-1]).all().item()):
            raise ValueError(f"Axis '{name}' must be strictly increasing.")
        axis_sizes.append(len(axis))

    expected_output_dims: tuple[int, ...] = (*axis_sizes, 1)

    if checkpoint.get("output_dims") != expected_output_dims:
        raise ValueError("Saved output dimensions do not match the map axes.")
    names = checkpoint["channel_names"]
    if not names or len(checkpoint["model_states"]) != len(names):
        raise ValueError("Channel/state count mismatch")

    parts: list[SubSurrogateModel] = []
    for state in checkpoint["model_states"]:
        mlp = MLP(
            num_input_neurons=len(checkpoint["feature_names"]),
            output_dims=tuple(checkpoint["output_dims"]),
            hidden_layer_neurons=tuple(checkpoint["hidden_layers"]),
        )
        mlp.load_state_dict(state, strict=True)
        mlp.eval()
        parts.append(SubSurrogateModel(mlp=mlp))
    return SurrogateModel(parts), checkpoint


@torch.no_grad()
def predict_raw_geometry(
    model: SurrogateModel,
    checkpoint: SurrogateCheckpoint,
    geometry: Tensor | Sequence[Sequence[float]],
) -> Tensor:
    values = torch.as_tensor(geometry, dtype=torch.float32).cpu().clone()
    names = checkpoint["feature_names"]

    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("Expected raw geometry [machines, features]")
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite geometry")

    for name, (minimum, maximum) in checkpoint["geometry_bounds"].items():
        index = names.index(name)
        values[:, index] = (values[:, index] - minimum) / (maximum - minimum)

    normalized = model.get_prediction(values)
    physical = (
        normalized * checkpoint["target_denominator"] + checkpoint["target_minimum"]
    )
    return physical
