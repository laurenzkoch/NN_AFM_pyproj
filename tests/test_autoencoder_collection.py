from pathlib import Path
from typing import cast

import pytest
import torch

from src.config.config_mcad_parameters import OUTPUT_NEURON_LIST
from src.modules.data_managment.experiment_utils import file_sha256
from src.modules.geometry.feature_schema import MACHINE_DEFINITION
from src.modules.neuronal_networks.autoencoder_collection import (
    AutoencoderCollection,
)
from src.modules.neuronal_networks.autoencoder_collection_trainer import (
    AutoencoderCollectionTrainer,
)
from src.modules.neuronal_networks.manipulating_data import (
    TrainValidationTestDataTensors,
)
from src.modules.neuronal_networks.model_data_io.autoencoder_io import (
    load_autoencoders,
    save_autoencoders,
)
from src.modules.neuronal_networks.model_data_io.evaluation_io import (
    EvaluationBundle,
    save_evaluation,
)
from src.modules.neuronal_networks.model_data_io.model_io import (
    load_bundle,
    save_bundle,
)
from src.modules.neuronal_networks.nn_training_torch import NNTrainer
from src.modules.neuronal_networks.NN_types.cnn_autoencoder import (
    Conv3dAutoencoder,
)
from src.modules.neuronal_networks.NN_types.mlp_with_torch import MLP
from src.modules.neuronal_networks.surrogate_model import (
    SubSurrogateModel,
    SurrogateModel,
    SurrogateModelTrainer,
)


@pytest.mark.training
def test_collection_training_and_reconstruction(
    tensor_data: TrainValidationTestDataTensors,
) -> None:
    torch.manual_seed(42)

    map_shape: tuple[int, int, int] = (
        tensor_data.train.map_tensor.shape[1],
        tensor_data.train.map_tensor.shape[2],
        tensor_data.train.map_tensor.shape[3],
    )
    networks: tuple[Conv3dAutoencoder, ...] = tuple(
        Conv3dAutoencoder(
            input_channels=1,
            features=(2, 4),
            latent_dim=2,
            map_shape=map_shape,
        )
        for _ in OUTPUT_NEURON_LIST
    )
    trainer: AutoencoderCollectionTrainer = AutoencoderCollectionTrainer(
        NNTrainer(num_epochs=2, patience=1)
    )
    collection: AutoencoderCollection = trainer.fit(
        autoencoders=networks,
        data=tensor_data,
    )

    assert collection.channel_names == tuple(OUTPUT_NEURON_LIST)
    assert len(collection.autoencoders) == len(OUTPUT_NEURON_LIST)
    assert len({id(network) for network in collection.autoencoders}) == len(
        OUTPUT_NEURON_LIST
    )

    for original, trained in zip(
        networks,
        collection.autoencoders,
        strict=True,
    ):
        assert trained is not original
        assert trained.input_channels == 1
        assert trained.features == (2, 4)
        assert trained.latent_dim == 2
        assert not trained.training
        assert all(not parameter.requires_grad for parameter in trained.parameters())

        duration: float | None = trained.training_duration_seconds
        assert duration is not None
        assert duration >= 0.0

    reference: torch.Tensor = tensor_data.validation.map_tensor
    original_reference: torch.Tensor = reference.clone()
    reconstructed: torch.Tensor = collection.reconstruct(reference)

    assert reconstructed.shape == reference.shape
    assert torch.isfinite(reconstructed).all().item()
    assert not reconstructed.requires_grad
    torch.testing.assert_close(reference, original_reference)


def test_collection_checkpoint_roundtrip(
    tmp_path: Path,
    tensor_data: TrainValidationTestDataTensors,
) -> None:
    torch.manual_seed(42)

    map_shape: tuple[int, int, int] = (
        tensor_data.train.map_tensor.shape[1],
        tensor_data.train.map_tensor.shape[2],
        tensor_data.train.map_tensor.shape[3],
    )
    networks: tuple[Conv3dAutoencoder, ...] = tuple(
        Conv3dAutoencoder(
            input_channels=1,
            features=(2, 4 + index),
            latent_dim=index + 1,
            map_shape=map_shape,
        )
        for index in range(len(OUTPUT_NEURON_LIST))
    )
    for index, network in enumerate(networks):
        network.training_duration_seconds = float(index + 1)

    collection: AutoencoderCollection = AutoencoderCollection(
        autoencoders=networks,
        map_axes={
            name: tuple(float(value) for value in axis)
            for name, axis in tensor_data.map_axes.items()
        },
        target_minimum=tensor_data.target_minimum.clone(),
        target_denominator=tensor_data.target_denominator.clone(),
        split_machine_ids=dict(tensor_data.split_machine_ids),
        source_hashes=dict(tensor_data.source_hashes),
    )
    collection.freeze()

    model_path: Path = tmp_path / "AutoencoderCollection.pt"
    save_autoencoders(
        path=model_path,
        collection=collection,
        metrics={"test": True},
        seed=42,
    )
    restored, checkpoint = load_autoencoders(model_path)

    assert checkpoint["model_type"] == "autoencoder_collection"
    assert checkpoint["seed"] == 42
    assert checkpoint["metrics"] == {"test": True}
    assert restored.channel_names == collection.channel_names
    assert restored.map_axes == collection.map_axes
    assert restored.split_machine_ids == collection.split_machine_ids
    assert restored.source_hashes == collection.source_hashes

    torch.testing.assert_close(
        restored.target_minimum,
        collection.target_minimum,
    )
    torch.testing.assert_close(
        restored.target_denominator,
        collection.target_denominator,
    )

    for original, loaded in zip(
        collection.autoencoders,
        restored.autoencoders,
        strict=True,
    ):
        assert loaded is not original
        assert loaded.features == original.features
        assert loaded.latent_dim == original.latent_dim
        assert loaded.training_duration_seconds == original.training_duration_seconds
        assert not loaded.training
        assert all(not parameter.requires_grad for parameter in loaded.parameters())

        for name, parameter in original.state_dict().items():
            torch.testing.assert_close(
                loaded.state_dict()[name],
                parameter,
                rtol=0,
                atol=0,
            )

    reference: torch.Tensor = tensor_data.validation.map_tensor
    torch.testing.assert_close(
        restored.reconstruct(reference),
        collection.reconstruct(reference),
    )

    # Es wird ausschließlich die gesamte Sammlung gespeichert.
    assert set(tmp_path.iterdir()) == {model_path}

    original_model_hash: str = file_sha256(model_path)

    evaluation_path: Path = save_evaluation(
        model_path=model_path,
        model=restored,
        data=tensor_data,
    )
    evaluation: EvaluationBundle = cast(
        EvaluationBundle,
        torch.load(
            evaluation_path,
            map_location="cpu",
            weights_only=True,
        ),
    )

    assert evaluation_path == tmp_path / "evaluation.pt"
    assert evaluation["format_version"] == 2
    assert evaluation["evaluation_kind"] == "reconstruction"
    assert tuple(evaluation["channel_names"]) == tuple(OUTPUT_NEURON_LIST)
    assert evaluation["model_filename"] == model_path.name
    assert evaluation["model_sha256"] == original_model_hash

    assert file_sha256(model_path) == original_model_hash
    assert set(tmp_path.iterdir()) == {
        model_path,
        evaluation_path,
    }


@pytest.mark.training
def test_combined_training_preserves_frozen_autoencoders(
    tmp_path: Path,
    tensor_data: TrainValidationTestDataTensors,
) -> None:
    torch.manual_seed(42)

    map_shape: tuple[int, int, int] = (
        tensor_data.train.map_tensor.shape[1],
        tensor_data.train.map_tensor.shape[2],
        tensor_data.train.map_tensor.shape[3],
    )
    networks: tuple[Conv3dAutoencoder, ...] = tuple(
        Conv3dAutoencoder(
            input_channels=1,
            features=(2, 4),
            latent_dim=index + 1,
            map_shape=map_shape,
        )
        for index in range(len(OUTPUT_NEURON_LIST))
    )
    cnn_trainer: AutoencoderCollectionTrainer = AutoencoderCollectionTrainer(
        NNTrainer(num_epochs=2, patience=1)
    )
    collection: AutoencoderCollection = cnn_trainer.fit(
        autoencoders=networks,
        data=tensor_data,
    )

    original_states: list[dict[str, torch.Tensor]] = [
        {name: value.detach().clone() for name, value in cnn.state_dict().items()}
        for cnn in collection.autoencoders
    ]
    original_durations: list[float | None] = [
        cnn.training_duration_seconds for cnn in collection.autoencoders
    ]

    model: SurrogateModel = SurrogateModel(
        submodels=[
            SubSurrogateModel(
                mlp=MLP(
                    len(MACHINE_DEFINITION),
                    (cnn.latent_dim,),
                    (4 + index,),
                )
            )
            for index, cnn in enumerate(collection.autoencoders)
        ]
    )
    trainer: SurrogateModelTrainer = SurrogateModelTrainer(
        NNTrainer(num_epochs=2, patience=1)
    )
    trained: SurrogateModel = trainer.fit(
        model=model,
        data=tensor_data,
        autoencoders=collection,
    )

    for index, part in enumerate(trained.submodels):
        cnn: Conv3dAutoencoder | None = part.cnn_autoencoder
        assert cnn is not None
        assert part.mlp.output_dims == (cnn.latent_dim,)
        assert part.mlp.training_duration_seconds is not None
        assert cnn.training_duration_seconds == original_durations[index]
        assert not cnn.training
        assert all(
            not parameter.requires_grad and parameter.grad is None
            for parameter in cnn.parameters()
        )

        for name, expected in original_states[index].items():
            torch.testing.assert_close(
                cnn.state_dict()[name],
                expected,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                collection.autoencoders[index].state_dict()[name],
                expected,
                rtol=0,
                atol=0,
            )

    prediction: torch.Tensor = trained.get_prediction(
        tensor_data.validation.geometry_tensor
    )
    assert prediction.shape == tensor_data.validation.map_tensor.shape
    assert torch.isfinite(prediction).all().item()

    model_path: Path = tmp_path / "SurrogateModel.pt"
    save_bundle(
        path=model_path,
        model=trained,
        data=tensor_data,
        metrics={},
        seed=42,
        autoencoders=collection,
    )
    restored, checkpoint = load_bundle(model_path)

    torch.testing.assert_close(
        restored.get_prediction(tensor_data.validation.geometry_tensor),
        prediction,
    )

    durations: list[float] = []
    for original_part, restored_part in zip(
        trained.submodels,
        restored.submodels,
        strict=True,
    ):
        assert (
            restored_part.mlp.hidden_layer_neurons
            == original_part.mlp.hidden_layer_neurons
        )
        assert restored_part.mlp.output_dims == original_part.mlp.output_dims

        original_cnn: Conv3dAutoencoder | None = original_part.cnn_autoencoder
        restored_cnn: Conv3dAutoencoder | None = restored_part.cnn_autoencoder
        assert original_cnn is not None
        assert restored_cnn is not None
        assert restored_cnn.latent_dim == original_cnn.latent_dim
        assert restored_cnn.features == original_cnn.features
        assert not restored_cnn.training
        assert all(
            not parameter.requires_grad for parameter in restored_cnn.parameters()
        )

        for original_network, restored_network in (
            (original_part.mlp, restored_part.mlp),
            (original_cnn, restored_cnn),
        ):
            duration: float | None = original_network.training_duration_seconds
            assert duration is not None
            assert restored_network.training_duration_seconds == duration
            durations.append(duration)

    assert checkpoint["training_duration_seconds"] == pytest.approx(sum(durations))
    assert checkpoint["autoencoder_provenance"] == {
        "split_machine_ids": collection.split_machine_ids,
        "source_hashes": collection.source_hashes,
    }
    assert set(tmp_path.iterdir()) == {model_path}
