import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, TypedDict

import torch
from torch import Tensor

from src.config.config_mcad_parameters import (
    AXIS_LIST,
    MACHINE_DEFINITION,
    OUTPUT_NEURON_LIST,
)
from src.modules.data_managment.experiment_utils import file_sha256
from src.modules.neuronal_networks.autoencoder_collection import (
    AutoencoderCollection,
)
from src.modules.neuronal_networks.manipulating_data import (
    MachineDataTensor,
    TrainValidationTestDataTensors,
)
from src.modules.neuronal_networks.model_data_io.model_io import JSONValue
from src.modules.neuronal_networks.surrogate_model import SurrogateModel
from src.modules.neuronal_networks.testing import (
    R2Report,
    calc_r_squared_report,
)


class EvaluationSplit(TypedDict):
    machine_ids: tuple[str, ...]
    geometry_inputs: Tensor
    prediction_physical: Tensor
    reference_physical: Tensor
    metrics: dict[str, JSONValue]


class EvaluationBundle(TypedDict):
    format_version: Literal[2]
    evaluation_kind: Literal["prediction", "reconstruction"]
    value_scale: Literal["physical"]

    model_filename: str
    model_sha256: str

    feature_names: list[str]
    channel_names: list[str]
    axis_names: list[str]
    map_axes: dict[str, list[float]]

    source_hashes: dict[str, str]
    splits: dict[Literal["train", "validation", "test"], EvaluationSplit]


@torch.no_grad()
def _build_evaluation_split(
    model: SurrogateModel | AutoencoderCollection,
    data: MachineDataTensor,
    machine_ids: tuple[str, ...],
    minimum: Tensor,
    denominator: Tensor,
) -> EvaluationSplit:
    geometry: Tensor = data.geometry_tensor.detach().cpu().clone()
    reference: Tensor = data.map_tensor.detach().cpu().double()

    if geometry.ndim != 2:
        raise ValueError("Expected geometry shape [machines, features].")

    if geometry.shape[1] != len(MACHINE_DEFINITION):
        raise ValueError("Unexpected number of geometry features.")

    if reference.ndim != 5:
        raise ValueError(
            "Expected map_reference_validation shape [machines, speed, d, q, channels]."
        )

    if (
        geometry.shape[0] == 0
        or geometry.shape[0] != reference.shape[0]
        or len(machine_ids) != geometry.shape[0]
    ):
        raise ValueError("Machine counts and machine IDs must match.")

    channel_count: int = len(OUTPUT_NEURON_LIST)

    if reference.shape[-1] != channel_count:
        raise ValueError("Unexpected number of output channels.")

    offset: Tensor = minimum.detach().cpu().double()
    scale: Tensor = denominator.detach().cpu().double()

    if offset.shape != (channel_count,) or scale.shape != (channel_count,):
        raise ValueError("Expected one normalization value per channel.")

    if not bool((scale > 0).all().item()):
        raise ValueError("Normalization denominators must be positive.")

    prediction: Tensor

    if isinstance(model, AutoencoderCollection):
        prediction = model.reconstruct(data.map_tensor).detach().cpu().double()
    else:
        for part in model.submodels:
            part.mlp.eval()

            if part.cnn_autoencoder is not None:
                part.cnn_autoencoder.eval()

        prediction = model.get_prediction(data.geometry_tensor).detach().cpu().double()

    if prediction.shape != reference.shape:
        raise ValueError("Prediction and map_reference_validation shapes must match.")

    for tensor in (geometry, reference, prediction, offset, scale):
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("Evaluation data contains non-finite values.")

    prediction_physical: Tensor = prediction * scale + offset
    reference_physical: Tensor = reference * scale + offset

    if not (
        bool(torch.isfinite(prediction_physical).all().item())
        and bool(torch.isfinite(reference_physical).all().item())
    ):
        raise ValueError("Physical evaluation values are non-finite.")

    normalized_errors: Tensor = (prediction - reference).reshape(-1, channel_count)
    physical_errors: Tensor = normalized_errors * scale

    r2: R2Report = calc_r_squared_report(reference, prediction)

    metrics: dict[str, JSONValue] = {
        "mse_normalized": normalized_errors.square().mean().item(),
        "r2_mean": r2.mean,
        "r2_per_channel": list(r2.per_channel),
        "mae_physical_per_channel": (physical_errors.abs().mean(dim=0).tolist()),
        "mse_physical_per_channel": (physical_errors.square().mean(dim=0).tolist()),
        "points_per_channel": normalized_errors.shape[0],
    }

    return {
        "machine_ids": machine_ids,
        "geometry_inputs": geometry,
        "prediction_physical": prediction_physical,
        "reference_physical": reference_physical,
        "metrics": metrics,
    }


def save_evaluation(
    model_path: Path,
    model: SurrogateModel | AutoencoderCollection,
    data: TrainValidationTestDataTensors,
) -> Path:

    if isinstance(model, AutoencoderCollection):
        model.validate_compatible(data)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    output_path: Path = model_path.with_name("evaluation.pt")
    if output_path.resolve() == model_path.resolve():
        raise ValueError("Evaluation path must differ from the model path.")

    model_hash: str = file_sha256(model_path)

    splits: dict[
        Literal["train", "validation", "test"],
        EvaluationSplit,
    ] = {
        "train": _build_evaluation_split(
            model=model,
            data=data.train,
            machine_ids=data.split_machine_ids["train"],
            minimum=data.target_minimum,
            denominator=data.target_denominator,
        ),
        "validation": _build_evaluation_split(
            model=model,
            data=data.validation,
            machine_ids=data.split_machine_ids["validation"],
            minimum=data.target_minimum,
            denominator=data.target_denominator,
        ),
    }

    if data.test is not None:
        splits["test"] = _build_evaluation_split(
            model=model,
            data=data.test,
            machine_ids=data.split_machine_ids["test"],
            minimum=data.target_minimum,
            denominator=data.target_denominator,
        )

    axis_names: list[str] = list(AXIS_LIST)
    if len(axis_names) != 3:
        raise ValueError("Expected three map axes.")

    map_axes: dict[str, list[float]] = {
        name: [float(value) for value in data.map_axes[name]] for name in axis_names
    }

    for split in splits.values():
        map_shape: tuple[int, ...] = tuple(split["reference_physical"].shape[1:-1])
        axis_shape: tuple[int, ...] = tuple(len(map_axes[name]) for name in axis_names)
        if map_shape != axis_shape:
            raise ValueError("Map axes do not match evaluation tensor shapes.")

    bundle: EvaluationBundle = {
        "format_version": 2,
        "evaluation_kind": (
            "reconstruction"
            if isinstance(model, AutoencoderCollection)
            else "prediction"
        ),
        "value_scale": "physical",
        "model_filename": model_path.name,
        "model_sha256": model_hash,
        "feature_names": list(MACHINE_DEFINITION),
        "channel_names": list(OUTPUT_NEURON_LIST),
        "axis_names": axis_names,
        "map_axes": map_axes,
        "source_hashes": dict(data.source_hashes),
        "splits": splits,
    }

    if file_sha256(model_path) != model_hash:
        raise RuntimeError("Model file changed during evaluation.")

    temporary_path: Path | None = None
    with NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.stem}_",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path: Path = Path(temporary_file.name)

    try:
        torch.save(bundle, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return output_path
