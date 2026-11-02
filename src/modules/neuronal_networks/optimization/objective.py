from __future__ import annotations

import copy
import random

import numpy as np
import optuna
import torch

from src.config.config_mcad_parameters import OUTPUT_NEURON_LIST
from src.modules.neuronal_networks.data_types import (
    MachineDataTensor,
    TrainValidationTestDataTensors,
)
from src.modules.neuronal_networks.nn_training_torch import NNTrainer
from src.modules.neuronal_networks.NN_types.cnn_autoencoder import (
    Conv3dAutoencoder,
)
from src.modules.neuronal_networks.NN_types.mlp_with_torch import MLP
from src.modules.neuronal_networks.optimization.optimization_types import (
    NetworkArchitecture,
    NetworkSearchSpace,
    NetworkType,
    checked_network_type,
)


class Objective:
    def __init__(
        self,
        data: TrainValidationTestDataTensors,
        trainer: NNTrainer,
        network_type: NetworkType,
        channel_index: int,
        search_space: NetworkSearchSpace,
        seed: int,
        trained_autoencoder: Conv3dAutoencoder | None = None,
    ) -> None:
        if network_type not in ("MLP", "CNN"):
            raise ValueError("network_type must be 'MLP' or 'CNN'.")

        if type(channel_index) is not int or not 0 <= channel_index < len(
            OUTPUT_NEURON_LIST
        ):
            raise ValueError("Invalid output channel index.")

        if network_type == "CNN" and trained_autoencoder is not None:
            raise ValueError("CNN optimization cannot receive a trained autoencoder.")

        self.data: TrainValidationTestDataTensors = data
        self.trainer: NNTrainer = trainer
        self.network_type: NetworkType = network_type
        self.channel_index: int = channel_index
        self.search_space: NetworkSearchSpace = search_space
        self.seed: int = seed

        self.trained_autoencoder: Conv3dAutoencoder | None = copy.deepcopy(
            trained_autoencoder
        )

        map_shape: tuple[int, ...] = tuple(data.train.map_tensor.shape[1:-1])

        if self.trained_autoencoder is not None:
            if self.trained_autoencoder.input_channels != 1:
                raise ValueError("The supplied CNN must process one output channel.")

            if tuple(self.trained_autoencoder.map_shape) != map_shape:
                raise ValueError("The supplied CNN has an incompatible map shape.")

    def __call__(self, trial: optuna.Trial) -> float:
        trial_seed: int = (self.seed + trial.number) % 2**32
        random.seed(trial_seed)
        np.random.seed(trial_seed)
        torch.manual_seed(trial_seed)

        layers: tuple[int, ...] = self._get_layer(trial)
        latent_dim: int | None = None

        if self.network_type == "CNN":
            latent_dim = trial.suggest_int(
                "latent_dim",
                self.search_space.min_latent_dim,
                self.search_space.max_latent_dim,
                step=self.search_space.latent_step,
            )

        architecture: NetworkArchitecture = NetworkArchitecture(
            network_type=checked_network_type(self.network_type),
            layers=layers,
            latent_dim=latent_dim,
            input_channels=1,
        )

        train_data: MachineDataTensor = self.data.train.select_channel(
            self.channel_index
        )
        validation_data: MachineDataTensor = self.data.validation.select_channel(
            self.channel_index
        )
        map_shape: tuple[int, ...] = tuple(train_data.map_tensor.shape[1:-1])

        autoencoder: Conv3dAutoencoder | None = copy.deepcopy(self.trained_autoencoder)
        network: MLP | Conv3dAutoencoder

        if architecture.network_type == "CNN":
            if architecture.latent_dim is None:
                raise ValueError("A CNN requires a latent dimension.")

            network = Conv3dAutoencoder(
                input_channels=architecture.input_channels,
                features=architecture.layers,
                latent_dim=architecture.latent_dim,
                map_shape=map_shape,
            )
        else:
            output_dims: tuple[int, ...] = (
                (*map_shape, 1) if autoencoder is None else (autoencoder.latent_dim,)
            )

            network = MLP(
                num_input_neurons=train_data.geometry_tensor.shape[1],
                output_dims=output_dims,
                hidden_layer_neurons=architecture.layers,
            )

        self.trainer.start(
            neuronal_network=network,
            train_data=train_data,
            validation_data=validation_data,
            trained_autoencoder=autoencoder,
        )

        validation_loss: float = float(self.trainer.best_validation_loss)

        if not np.isfinite(validation_loss):
            raise FloatingPointError("The trial produced a non-finite validation loss.")

        trial.set_user_attr(
            "architecture",
            {
                "network_type": architecture.network_type,
                "layers": list(architecture.layers),
                "latent_dim": architecture.latent_dim,
                "input_channels": architecture.input_channels,
            },
        )
        trial.set_user_attr("channel_index", self.channel_index)
        trial.set_user_attr(
            "loss_scope",
            (
                "small_map"
                if architecture.network_type == "MLP" and autoencoder is not None
                else "full_map"
            ),
        )

        return validation_loss

    def _get_layer(self, trial: optuna.Trial) -> tuple[int, ...]:

        choices: tuple[int, ...] = self.search_space.unit_choices()

        if self.network_type == "MLP":
            count: int = trial.suggest_int(
                "n_layers",
                1,
                self.search_space.max_num_layers,
            )

            return tuple(
                int(
                    trial.suggest_categorical(
                        f"neurons_{index}",
                        choices,
                    )
                )
                for index in range(count)
            )

        layer_limit: int = min(
            self.search_space.max_num_layers,
            len(choices),
        )
        count = trial.suggest_int("n_layers", 1, layer_limit)

        features: list[int] = []
        previous_index: int = -1

        for layer_index in range(count):
            # Reserve sufficient larger feature counts for later layers.
            maximum_index: int = len(choices) - (count - layer_index)

            feature_index: int = trial.suggest_int(
                f"feature_index_{layer_index}",
                previous_index + 1,
                maximum_index,
            )

            features.append(choices[feature_index])
            previous_index = feature_index

        return tuple(features)
