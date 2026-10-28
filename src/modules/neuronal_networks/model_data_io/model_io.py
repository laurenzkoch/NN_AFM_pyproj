from __future__ import annotations

import logging
import math
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
    OUTPUT_NEURON_LIST,
)
from src.modules.geometry.feature_schema import MACHINE_DEFINITION
from src.modules.neuronal_networks.autoencoder_collection import (
    AutoencoderCollection,
)
from src.modules.neuronal_networks.manipulating_data import (
    TrainValidationTestDataTensors,
)
from src.modules.neuronal_networks.model_data_io.network_io import (
    AutoencoderCheckpoint,
    MLPCheckpoint,
    checked_training_duration,
    pack_autoencoder,
    pack_mlp,
    unpack_autoencoder,
    unpack_mlp,
)
from src.modules.neuronal_networks.NN_types.cnn_autoencoder import (
    Conv3dAutoencoder,
)
from src.modules.neuronal_networks.NN_types.mlp_with_torch import MLP
from src.modules.neuronal_networks.surrogate_model import (
    SubSurrogateModel,
    SurrogateModel,
)

type JSONValue = (
    str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]
)

logger = logging.getLogger(__name__)


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


class SubmodelCheckpoint(TypedDict):
    channel_name: str
    mlp: MLPCheckpoint
    cnn_autoencoder: AutoencoderCheckpoint | None


class AutoencoderProvenance(TypedDict):
    split_machine_ids: dict[str, tuple[str, ...]]
    source_hashes: dict[str, str]


class SurrogateCheckpointV2(TypedDict):
    format_version: Literal[2]
    model_type: Literal["surrogate_model"]
    feature_names: list[str]
    channel_names: list[str]
    geometry_bounds: dict[str, tuple[float, float]]
    map_axes: dict[str, list[float]]
    axis_names: list[str]
    output_dims: tuple[int, ...]
    target_minimum: Tensor
    target_denominator: Tensor
    split_machine_ids: dict[str, tuple[str, ...]]
    source_hashes: dict[str, str]
    autoencoder_provenance: AutoencoderProvenance | None
    seed: int
    metrics: dict[str, JSONValue]
    training_duration_seconds: float | None
    submodels: list[SubmodelCheckpoint]


def save_bundle(
    path: Path,
    model: SurrogateModel,
    data: TrainValidationTestDataTensors,
    metrics: dict[str, JSONValue],
    seed: int,
    autoencoders: AutoencoderCollection | None = None,
) -> None:

    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid seed.")

    if len(model.submodels) != len(OUTPUT_NEURON_LIST):
        raise ValueError("Wrong number of channel models.")

    has_cnn: bool = any(part.cnn_autoencoder is not None for part in model.submodels)

    if has_cnn and autoencoders is None:
        raise ValueError(
            "Provide the CNN collection to preserve its training provenance."
        )

    if not has_cnn and autoencoders is not None:
        raise ValueError("A direct MLP surrogate must not receive CNN provenance.")

    if autoencoders is not None:
        autoencoders.validate_compatible(data)

    output_dims: tuple[int, ...] = (
        *data.train.map_tensor.shape[1:-1],
        1,
    )

    submodels: list[SubmodelCheckpoint] = []
    durations: list[float | None] = []

    for channel_name, part in zip(
        OUTPUT_NEURON_LIST,
        model.submodels,
        strict=True,
    ):
        if part.mlp.num_input_neurons != len(MACHINE_DEFINITION):
            raise ValueError(f"Wrong MLP input size: {channel_name}")

        mlp_checkpoint: MLPCheckpoint = pack_mlp(part.mlp)
        cnn_checkpoint: AutoencoderCheckpoint | None = None

        expected_mlp_output: tuple[int, ...] = output_dims

        if part.cnn_autoencoder is not None:
            cnn = part.cnn_autoencoder

            if cnn.input_channels != 1 or tuple(cnn.map_shape) != output_dims[:-1]:
                raise ValueError(f"Invalid CNN dimensions: {channel_name}")

            expected_mlp_output = (cnn.latent_dim,)
            packed_cnn: AutoencoderCheckpoint = pack_autoencoder(cnn)

            if autoencoders is None:
                raise ValueError("Missing CNN collection.")

            reference_cnn = autoencoders.for_channel(channel_name)
            reference_state: dict[str, Tensor] = dict(reference_cnn.state_dict())
            saved_state: dict[str, Tensor] = packed_cnn["state_dict"]

            if (
                cnn.features != reference_cnn.features
                or cnn.latent_dim != reference_cnn.latent_dim
                or cnn.kernel_size != reference_cnn.kernel_size
                or set(saved_state) != set(reference_state)
            ):
                raise ValueError(
                    f"CNN differs from its source collection: {channel_name}"
                )

            for name, value in saved_state.items():
                if not torch.equal(
                    value,
                    reference_state[name].detach().cpu(),
                ):
                    raise ValueError(
                        f"CNN weights differ from their source: {channel_name}, {name}"
                    )

            durations.append(packed_cnn["training_duration_seconds"])
            cnn_checkpoint = packed_cnn

        if part.mlp.output_dims != expected_mlp_output:
            raise ValueError(f"Wrong MLP output dimensions: {channel_name}")

        durations.append(mlp_checkpoint["training_duration_seconds"])
        submodels.append(
            {
                "channel_name": channel_name,
                "mlp": mlp_checkpoint,
                "cnn_autoencoder": cnn_checkpoint,
            }
        )

    total_duration: float | None = None

    if all(duration is not None for duration in durations):
        total_duration = checked_training_duration(
            sum(duration for duration in durations if duration is not None)
        )

    provenance: AutoencoderProvenance | None = None

    if autoencoders is not None:
        provenance = {
            "split_machine_ids": {
                name: tuple(machine_ids)
                for name, machine_ids in autoencoders.split_machine_ids.items()
            },
            "source_hashes": dict(autoencoders.source_hashes),
        }

    checkpoint: SurrogateCheckpointV2 = {
        "format_version": 2,
        "model_type": "surrogate_model",
        "feature_names": list(MACHINE_DEFINITION),
        "channel_names": list(OUTPUT_NEURON_LIST),
        "geometry_bounds": dict(GEO_PARAMETERS_MIN_MAX_DICT),
        "map_axes": {name: values.tolist() for name, values in data.map_axes.items()},
        "axis_names": list(AXIS_LIST),
        "output_dims": output_dims,
        "target_minimum": data.target_minimum.detach().cpu().clone(),
        "target_denominator": (data.target_denominator.detach().cpu().clone()),
        "split_machine_ids": {
            name: tuple(machine_ids)
            for name, machine_ids in data.split_machine_ids.items()
        },
        "source_hashes": dict(data.source_hashes),
        "autoencoder_provenance": provenance,
        "seed": seed,
        "metrics": metrics,
        "training_duration_seconds": total_duration,
        "submodels": submodels,
    }

    output_path: Path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=".surrogate_",
        suffix=".pt",
        delete=False,
    ) as stream:
        temporary_path: Path = Path(stream.name)

    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    logger.info(
        "Saved complete surrogate: %s, training duration: %s s",
        output_path,
        total_duration,
    )


def _validate_surrogate_metadata(
    checkpoint: SurrogateCheckpointV2,
) -> None:

    if checkpoint["feature_names"] != list(MACHINE_DEFINITION):
        raise ValueError(
            "Saved geometry features differ from the current configuration."
        )

    if checkpoint["channel_names"] != list(OUTPUT_NEURON_LIST):
        raise ValueError("Saved output channels differ from the current configuration.")

    if checkpoint["axis_names"] != list(AXIS_LIST):
        raise ValueError("Saved axis names differ from the current configuration.")

    channel_count: int = len(OUTPUT_NEURON_LIST)

    for name in ("target_minimum", "target_denominator"):
        value: object = checkpoint[name]

        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a tensor.")

        if (
            value.shape != (channel_count,)
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all().item())
        ):
            raise ValueError(
                f"{name} must contain one finite floating-point "
                "value per output channel."
            )

        if name == "target_denominator":
            if not bool((value > 0).all().item()):
                raise ValueError("target_denominator must be strictly positive.")

    axes: object = checkpoint["map_axes"]

    if not isinstance(axes, dict) or set(axes) != set(AXIS_LIST):
        raise ValueError("Missing or unexpected map axes.")

    axis_sizes: list[int] = []

    for axis_name in AXIS_LIST:
        coordinates: object = axes[axis_name]

        if (
            not isinstance(coordinates, list)
            or not coordinates
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in coordinates
            )
        ):
            raise ValueError(f"Invalid coordinates for axis: {axis_name}")

        axis_values: Tensor = torch.tensor(
            coordinates,
            dtype=torch.float64,
        )

        if not bool(torch.isfinite(axis_values).all().item()) or not bool(
            (axis_values[1:] > axis_values[:-1]).all().item()
        ):
            raise ValueError(
                f"Axis must be finite and strictly increasing: {axis_name}"
            )

        axis_sizes.append(len(coordinates))

    if checkpoint["output_dims"] != (*axis_sizes, 1):
        raise ValueError("Saved output dimensions do not match the map axes.")

    if type(checkpoint["seed"]) is not int or not 0 <= checkpoint["seed"] < 2**32:
        raise ValueError("Invalid saved seed.")

    if not isinstance(checkpoint["metrics"], dict):
        raise TypeError("metrics must contain a dictionary.")

    checked_training_duration(checkpoint["training_duration_seconds"])


def load_bundle(
    path: Path,
) -> tuple[SurrogateModel, SurrogateCheckpointV2]:
    """Restore a complete version-2 surrogate and its metadata."""
    raw: object = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(raw, dict):
        raise TypeError("The model bundle must contain a dictionary.")

    required_fields: set[str] = set(SurrogateCheckpointV2.__annotations__)
    missing_fields: set[str] = required_fields.difference(raw)

    if missing_fields:
        raise ValueError(f"Missing checkpoint fields: {sorted(missing_fields)}")

    if raw.get("format_version") != 2:
        raise ValueError("Only surrogate format version 2 is supported.")

    if raw.get("model_type") != "surrogate_model":
        raise ValueError("Expected a complete surrogate model.")

    checkpoint: SurrogateCheckpointV2 = cast(
        SurrogateCheckpointV2,
        raw,
    )
    _validate_surrogate_metadata(checkpoint)

    raw_submodels: object = raw.get("submodels")

    if not isinstance(raw_submodels, list) or len(raw_submodels) != len(
        OUTPUT_NEURON_LIST
    ):
        raise ValueError("Wrong number of saved channel models.")

    parts: list[SubSurrogateModel] = []
    durations: list[float | None] = []
    has_cnn: bool = False

    for channel_name, raw_part in zip(
        OUTPUT_NEURON_LIST,
        raw_submodels,
        strict=True,
    ):
        if not isinstance(raw_part, dict):
            raise TypeError(f"Invalid submodel entry: {channel_name}")

        if raw_part.get("channel_name") != channel_name:
            raise ValueError(f"Wrong channel assignment: {channel_name}")

        if not {"mlp", "cnn_autoencoder"}.issubset(raw_part):
            raise ValueError(f"Incomplete submodel entry: {channel_name}")

        mlp: MLP = unpack_mlp(raw_part["mlp"])
        cnn: Conv3dAutoencoder | None = None

        expected_output: tuple[int, ...] = checkpoint["output_dims"]
        raw_cnn: object = raw_part["cnn_autoencoder"]

        if raw_cnn is not None:
            loaded_cnn: Conv3dAutoencoder = unpack_autoencoder(raw_cnn)

            if (
                loaded_cnn.input_channels != 1
                or tuple(loaded_cnn.map_shape) != checkpoint["output_dims"][:-1]
            ):
                raise ValueError(f"Invalid CNN dimensions: {channel_name}")

            loaded_cnn.requires_grad_(False)
            expected_output = (loaded_cnn.latent_dim,)
            durations.append(loaded_cnn.training_duration_seconds)

            cnn = loaded_cnn
            has_cnn = True

        if mlp.num_input_neurons != len(checkpoint["feature_names"]):
            raise ValueError(f"Wrong MLP input size: {channel_name}")

        if mlp.output_dims != expected_output:
            raise ValueError(f"Wrong MLP output dimensions: {channel_name}")

        durations.append(mlp.training_duration_seconds)
        parts.append(
            SubSurrogateModel(
                mlp=mlp,
                cnn_autoencoder=cnn,
            )
        )

    provenance: object = checkpoint["autoencoder_provenance"]

    if has_cnn:
        if (
            not isinstance(provenance, dict)
            or not isinstance(provenance.get("split_machine_ids"), dict)
            or not isinstance(provenance.get("source_hashes"), dict)
        ):
            raise ValueError("Missing or invalid CNN training provenance.")
    elif provenance is not None:
        raise ValueError("A direct MLP surrogate must not contain CNN provenance.")

    expected_duration: float | None = None

    if all(duration is not None for duration in durations):
        expected_duration = checked_training_duration(
            sum(duration for duration in durations if duration is not None)
        )

    saved_duration: float | None = checked_training_duration(
        checkpoint["training_duration_seconds"]
    )

    if expected_duration is None:
        if saved_duration is not None:
            raise ValueError(
                "A total training duration is stored although "
                "individual training durations are unknown."
            )
    elif saved_duration is None or not math.isclose(
        saved_duration,
        expected_duration,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "Total training duration differs from the individual durations."
        )

    model: SurrogateModel = SurrogateModel(parts)

    logger.info(
        "Loaded complete surrogate: %s, training duration: %s s",
        path,
        saved_duration,
    )

    return model, checkpoint


@torch.no_grad()
def predict_raw_geometry(
    model: SurrogateModel,
    checkpoint: SurrogateCheckpointV2,
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
