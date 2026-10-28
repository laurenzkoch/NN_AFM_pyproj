from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch

from src.config.config_mcad_parameters import (
    MAP_SIZE,
    OUTPUT_NEURON_LIST,
)
from src.modules.data_managment.experiment_utils import file_sha256
from src.modules.geometry.feature_schema import MACHINE_DEFINITION
from src.modules.neuronal_networks.manipulating_data import (
    MachineDataTensor,
    TrainValidationTestDataTensors,
)
from src.modules.neuronal_networks.model_data_io.evaluation_io import (
    EvaluationBundle,
    save_evaluation,
)
from src.modules.neuronal_networks.model_data_io.model_io import (
    load_bundle,
    predict_raw_geometry,
    save_bundle,
)
from src.modules.neuronal_networks.nn_training_torch import NNTrainer
from src.modules.neuronal_networks.NN_types.mlp_with_torch import MLP
from src.modules.neuronal_networks.surrogate_model import (
    SubSurrogateModel,
    SurrogateModel,
    SurrogateModelTrainer,
)
from src.modules.neuronal_networks.testing import evaluate_surrogate


def make_model() -> SurrogateModel:
    torch.manual_seed(100)
    return SurrogateModel(
        [
            SubSurrogateModel(MLP(len(MACHINE_DEFINITION), (*MAP_SIZE, 1), (4,)))
            for _ in OUTPUT_NEURON_LIST
        ]
    )


def test_all_seven_submodels_survive_bundle_roundtrip(
    tmp_path: Path, tensor_data: TrainValidationTestDataTensors
) -> None:
    model = make_model()
    path = tmp_path / "model" / "surrogate.pt"
    save_bundle(
        path=path,
        model=model,
        data=tensor_data,
        metrics={"test": True},
        seed=42,
    )
    loaded, checkpoint = load_bundle(path)
    assert len(loaded.submodels) == 7
    assert checkpoint["metrics"] == {"test": True}
    assert checkpoint["source_hashes"] == tensor_data.source_hashes
    assert checkpoint["split_machine_ids"] == tensor_data.split_machine_ids
    torch.testing.assert_close(
        model.get_prediction(tensor_data.validation.geometry_tensor),
        loaded.get_prediction(tensor_data.validation.geometry_tensor),
    )
    assert all(not part.mlp.training for part in loaded.submodels)


@pytest.mark.parametrize(
    "field",
    [
        "feature_names",
        "channel_names",
        "target_denominator",
        "axis_names",
        "output_dims",
    ],
)
def test_loading_rejects_incompatible_bundle_fields(
    tmp_path: Path, tensor_data: TrainValidationTestDataTensors, field: str
) -> None:
    path = tmp_path / "surrogate.pt"
    save_bundle(
        path=path,
        model=make_model(),
        data=tensor_data,
        metrics={},
        seed=42,
    )
    document = torch.load(path, map_location="cpu", weights_only=True)
    replacements: dict[str, object] = {
        "feature_names": ["wrong"],
        "channel_names": ["wrong"],
        "target_denominator": torch.zeros(len(OUTPUT_NEURON_LIST)),
        "axis_names": ["wrong"],
        "output_dims": (1, 1, 1, 1),
    }
    document[field] = replacements[field]
    torch.save(document, path)
    with pytest.raises(ValueError):
        load_bundle(path)


def test_raw_prediction_does_not_modify_input(
    tmp_path: Path, tensor_data: TrainValidationTestDataTensors
) -> None:
    path = tmp_path / "surrogate.pt"
    save_bundle(
        path=path,
        model=make_model(),
        data=tensor_data,
        metrics={},
        seed=42,
    )
    model, checkpoint = load_bundle(path)
    values = torch.ones((2, len(MACHINE_DEFINITION)))
    for name, (minimum, maximum) in checkpoint["geometry_bounds"].items():
        values[:, checkpoint["feature_names"].index(name)] = (minimum + maximum) / 2
    original = values.clone()
    prediction = predict_raw_geometry(model, checkpoint, values)
    assert prediction.shape == (2, *MAP_SIZE, 7)
    assert torch.isfinite(prediction).all()
    torch.testing.assert_close(values, original)


@pytest.mark.training
def test_training_produces_independent_channel_models(
    tensor_data: TrainValidationTestDataTensors,
) -> None:
    torch.manual_seed(42)
    model: SurrogateModel = make_model()

    trainer: SurrogateModelTrainer = SurrogateModelTrainer(
        NNTrainer(num_epochs=2, patience=1)
    )
    trained: SurrogateModel = trainer.fit(
        model=model,
        data=tensor_data,
    )

    assert trained is model
    assert len(trained.submodels) == tensor_data.train.map_tensor.shape[-1]

    networks: list[MLP] = [part.mlp for part in trained.submodels]
    assert len({id(network) for network in networks}) == len(networks)

    # Verschiedene Modelle dürfen keinen Parameterspeicher teilen.
    parameter_addresses: list[set[int]] = [
        {parameter.data_ptr() for parameter in network.parameters()}
        for network in networks
    ]
    for index, addresses in enumerate(parameter_addresses):
        for other in parameter_addresses[index + 1 :]:
            assert addresses.isdisjoint(other)

    for network in networks:
        duration: float | None = network.training_duration_seconds
        assert duration is not None
        assert duration >= 0.0
        assert not network.training

    prediction: torch.Tensor = trained.get_prediction(
        tensor_data.validation.geometry_tensor
    )
    assert prediction.shape == tensor_data.validation.map_tensor.shape
    assert torch.isfinite(prediction).all().item()


@pytest.mark.parametrize(
    "learning_rate,weight_decay,num_epochs,patience",
    [
        (0.0, 0.0, 2, 1),
        (float("nan"), 0.0, 2, 1),
        (0.001, -1.0, 2, 1),
        (0.001, 0.0, True, 1),
        (0.001, 0.0, 2, 0),
    ],
)
def test_trainer_rejects_invalid_settings(
    learning_rate: float, weight_decay: float, num_epochs: int, patience: int
) -> None:
    with pytest.raises(ValueError):
        NNTrainer(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            patience=patience,
        )


@pytest.mark.parametrize("include_test", [False, True])
def test_evaluation_export(
    tmp_path: Path,
    tensor_data: TrainValidationTestDataTensors,
    include_test: bool,
) -> None:
    channel_count: int = len(OUTPUT_NEURON_LIST)

    test_data: MachineDataTensor | None = (
        MachineDataTensor(
            geometry_tensor=tensor_data.validation.geometry_tensor.clone(),
            map_tensor=tensor_data.validation.map_tensor.clone() + 0.25,
        )
        if include_test
        else None
    )

    data: TrainValidationTestDataTensors = replace(
        tensor_data,
        test=test_data,
        target_minimum=torch.arange(1, channel_count + 1, dtype=torch.float64),
        target_denominator=torch.arange(2, channel_count + 2, dtype=torch.float64),
        split_machine_ids={
            **tensor_data.split_machine_ids,
            "test": ("test-0",) if include_test else (),
        },
    )

    model: SurrogateModel = make_model()
    model_path: Path = tmp_path / "surrogate.pt"
    save_bundle(
        path=model_path,
        model=model,
        data=data,
        metrics={},
        seed=42,
    )
    loaded, _ = load_bundle(model_path)

    original_model_hash: str = file_sha256(model_path)
    output_path: Path = save_evaluation(model_path, loaded, data)

    bundle: EvaluationBundle = cast(
        EvaluationBundle,
        torch.load(output_path, map_location="cpu", weights_only=True),
    )

    assert output_path == tmp_path / "evaluation.pt"
    assert bundle["format_version"] == 2
    assert bundle["evaluation_kind"] == "prediction"
    assert bundle["value_scale"] == "physical"
    assert bundle["model_filename"] == model_path.name
    assert bundle["model_sha256"] == original_model_hash
    assert file_sha256(model_path) == original_model_hash
    assert bundle["feature_names"] == list(MACHINE_DEFINITION)
    assert bundle["channel_names"] == list(OUTPUT_NEURON_LIST)
    assert bundle["source_hashes"] == data.source_hashes
    assert bundle["axis_names"] == list(data.map_axes)
    assert bundle["map_axes"] == {
        name: axis.tolist() for name, axis in data.map_axes.items()
    }
    assert set(bundle["splits"]) == (
        {"train", "validation", "test"} if include_test else {"train", "validation"}
    )

    for name, split in bundle["splits"].items():
        source: MachineDataTensor | None

        if name == "train":
            source = data.train
        elif name == "validation":
            source = data.validation
        else:
            source = data.test
        assert source is not None

        expected_metrics = evaluate_surrogate(
            model=loaded,
            data=source,
            denominator=data.target_denominator,
        )

        assert set(split["metrics"]) == set(expected_metrics)

        for metric_name, expected_value in expected_metrics.items():
            actual_value = split["metrics"][metric_name]

            if expected_value is None:
                assert actual_value is None
            else:
                assert actual_value == pytest.approx(expected_value)

        with torch.no_grad():
            expected_prediction = (
                loaded.get_prediction(source.geometry_tensor).double()
                * data.target_denominator
                + data.target_minimum
            )

        expected_reference = (
            source.map_tensor.double() * data.target_denominator + data.target_minimum
        )

        assert split["machine_ids"] == data.split_machine_ids[name]
        assert split["prediction_physical"].shape == source.map_tensor.shape

        torch.testing.assert_close(split["geometry_inputs"], source.geometry_tensor)
        torch.testing.assert_close(split["prediction_physical"], expected_prediction)
        torch.testing.assert_close(split["reference_physical"], expected_reference)

        for field in (
            "geometry_inputs",
            "prediction_physical",
            "reference_physical",
        ):
            value = split[field]
            assert value.device.type == "cpu"
            assert not value.requires_grad

    assert set(tmp_path.iterdir()) == {model_path, output_path}
