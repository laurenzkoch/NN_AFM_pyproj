import logging
import os
import tempfile
from pathlib import Path
from typing import Literal, TypedDict, cast

import torch
from torch import Tensor

from src.config.config_mcad_parameters import AXIS_LIST, OUTPUT_NEURON_LIST
from src.modules.neuronal_networks.autoencoder_collection import (
    AutoencoderCollection,
)
from src.modules.neuronal_networks.model_data_io.model_io import JSONValue
from src.modules.neuronal_networks.model_data_io.network_io import (
    AutoencoderCheckpoint,
    pack_autoencoder,
    unpack_autoencoder,
)

logger = logging.getLogger(__name__)


class AutoencoderCollectionCheckpoint(TypedDict):
    format_version: Literal[1]
    model_type: Literal["autoencoder_collection"]
    channel_names: list[str]
    axis_names: list[str]
    map_axes: dict[str, list[float]]
    target_minimum: Tensor
    target_denominator: Tensor
    split_machine_ids: dict[str, tuple[str, ...]]
    source_hashes: dict[str, str]
    seed: int
    metrics: dict[str, JSONValue]
    autoencoders: list[AutoencoderCheckpoint]


def save_autoencoders(
    path: Path,
    collection: AutoencoderCollection,
    metrics: dict[str, JSONValue],
    seed: int,
) -> None:

    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid seed.")

    if set(collection.map_axes) != set(AXIS_LIST):
        raise ValueError("Missing or unexpected map axes.")

    map_shape: tuple[int, ...] = tuple(
        len(collection.map_axes[name]) for name in AXIS_LIST
    )

    for channel_name, autoencoder in zip(
        collection.channel_names,
        collection.autoencoders,
        strict=True,
    ):
        if tuple(autoencoder.map_shape) != map_shape:
            raise ValueError(
                f"Autoencoder dimensions differ from the map axes: {channel_name}"
            )

    checkpoint: AutoencoderCollectionCheckpoint = {
        "format_version": 1,
        "model_type": "autoencoder_collection",
        "channel_names": list(collection.channel_names),
        "axis_names": list(AXIS_LIST),
        "map_axes": {
            name: [float(value) for value in collection.map_axes[name]]
            for name in AXIS_LIST
        },
        "target_minimum": (collection.target_minimum.detach().cpu().clone()),
        "target_denominator": (collection.target_denominator.detach().cpu().clone()),
        "split_machine_ids": {
            name: tuple(machine_ids)
            for name, machine_ids in collection.split_machine_ids.items()
        },
        "source_hashes": dict(collection.source_hashes),
        "seed": seed,
        "metrics": metrics,
        "autoencoders": [
            pack_autoencoder(autoencoder) for autoencoder in collection.autoencoders
        ],
    }

    output_path: Path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=".autoencoders_",
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
        "Saved %d channel autoencoders: %s",
        len(collection.autoencoders),
        output_path,
    )


def load_autoencoders(
    path: Path,
) -> tuple[AutoencoderCollection, AutoencoderCollectionCheckpoint]:

    raw: object = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(raw, dict):
        raise TypeError("The autoencoder bundle must contain a dictionary.")

    required_fields: set[str] = set(AutoencoderCollectionCheckpoint.__annotations__)
    missing_fields: set[str] = required_fields.difference(raw)

    if missing_fields:
        raise ValueError(f"Missing autoencoder bundle fields: {sorted(missing_fields)}")

    if raw.get("format_version") != 1:
        raise ValueError("Unsupported autoencoder bundle version.")

    if raw.get("model_type") != "autoencoder_collection":
        raise ValueError("Expected an autoencoder collection.")

    if raw.get("channel_names") != list(OUTPUT_NEURON_LIST):
        raise ValueError(
            "Saved channel names or their order differ from the current configuration."
        )

    if raw.get("axis_names") != list(AXIS_LIST):
        raise ValueError(
            "Saved axis names or their order differ from the current configuration."
        )

    raw_axes: object = raw.get("map_axes")

    if not isinstance(raw_axes, dict) or set(raw_axes) != set(AXIS_LIST):
        raise ValueError("Missing or unexpected map axes.")

    map_axes: dict[str, tuple[float, ...]] = {}

    for axis_name in AXIS_LIST:
        raw_axis: object = raw_axes[axis_name]

        if (
            not isinstance(raw_axis, list)
            or not raw_axis
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in raw_axis
            )
        ):
            raise ValueError(f"Invalid coordinates for axis: {axis_name}")

        axis_values: Tensor = torch.tensor(
            raw_axis,
            dtype=torch.float64,
        )

        if not bool(torch.isfinite(axis_values).all().item()) or not bool(
            (axis_values[1:] > axis_values[:-1]).all().item()
        ):
            raise ValueError(
                f"Axis must be finite and strictly increasing: {axis_name}"
            )

        map_axes[axis_name] = tuple(float(value) for value in raw_axis)

    raw_models: object = raw.get("autoencoders")

    if not isinstance(raw_models, list) or len(raw_models) != len(OUTPUT_NEURON_LIST):
        raise ValueError("Wrong number of saved autoencoders.")

    minimum: object = raw.get("target_minimum")
    denominator: object = raw.get("target_denominator")

    if not isinstance(minimum, Tensor) or not isinstance(denominator, Tensor):
        raise TypeError("Normalization values must be tensors.")

    raw_splits: object = raw.get("split_machine_ids")

    if not isinstance(raw_splits, dict):
        raise TypeError("split_machine_ids must contain a dictionary.")

    split_ids: dict[str, tuple[str, ...]] = {}

    for split_name, machine_ids in raw_splits.items():
        if (
            not isinstance(split_name, str)
            or not isinstance(machine_ids, (list, tuple))
            or any(not isinstance(value, str) for value in machine_ids)
        ):
            raise TypeError("Invalid dataset split information.")

        if len(set(machine_ids)) != len(machine_ids):
            raise ValueError(f"Duplicate machine IDs in split: {split_name}")

        split_ids[split_name] = tuple(machine_ids)

    if not {"train", "validation"}.issubset(split_ids):
        raise ValueError("Training or validation machine IDs are missing.")

    if set(split_ids["train"]) & set(split_ids["validation"]):
        raise ValueError("CNN training and validation machines overlap.")

    raw_sources: object = raw.get("source_hashes")

    if not isinstance(raw_sources, dict) or any(
        not isinstance(name, str) or not isinstance(digest, str)
        for name, digest in raw_sources.items()
    ):
        raise TypeError("Invalid source hashes.")

    if not isinstance(raw.get("metrics"), dict):
        raise TypeError("metrics must contain a dictionary.")

    saved_seed: object = raw.get("seed")

    if type(saved_seed) is not int or not 0 <= saved_seed < 2**32:
        raise ValueError("Invalid saved seed.")

    collection: AutoencoderCollection = AutoencoderCollection(
        autoencoders=tuple(unpack_autoencoder(model) for model in raw_models),
        map_axes=map_axes,
        target_minimum=minimum.detach().cpu().clone(),
        target_denominator=denominator.detach().cpu().clone(),
        split_machine_ids=split_ids,
        source_hashes=dict(raw_sources),
    )

    expected_shape: tuple[int, ...] = tuple(len(map_axes[name]) for name in AXIS_LIST)

    for channel_name, autoencoder in zip(
        collection.channel_names,
        collection.autoencoders,
        strict=True,
    ):
        if tuple(autoencoder.map_shape) != expected_shape:
            raise ValueError(
                f"Saved CNN dimensions differ from the map axes: {channel_name}"
            )

    collection.freeze()

    checkpoint: AutoencoderCollectionCheckpoint = cast(
        AutoencoderCollectionCheckpoint,
        raw,
    )

    logger.info(
        "Loaded %d channel autoencoders: %s",
        len(collection.autoencoders),
        path,
    )

    return collection, checkpoint
