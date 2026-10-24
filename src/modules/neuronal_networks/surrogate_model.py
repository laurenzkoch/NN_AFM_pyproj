import copy
import logging
from dataclasses import dataclass

import torch
from torch import Tensor

from src.config.config_mcad_parameters import OUTPUT_NEURON_LIST
from src.modules.neuronal_networks.autoencoder_collection import (
    AutoencoderCollection,
)
from src.modules.neuronal_networks.manipulating_data import (
    TrainValidationTestDataTensors,
)
from src.modules.neuronal_networks.nn_training_torch import NNTrainer
from src.modules.neuronal_networks.NN_types.cnn_autoencoder import Conv3dAutoencoder
from src.modules.neuronal_networks.NN_types.mlp_with_torch import MLP

logger = logging.getLogger(__name__)


@dataclass
class SubSurrogateModel:
    mlp: MLP
    cnn_autoencoder: Conv3dAutoencoder | None = None


class SurrogateModel:
    def __init__(self, submodels: list[SubSurrogateModel]) -> None:
        self.submodels = submodels

    @torch.no_grad()
    def get_prediction(self, geo_tensor: Tensor) -> Tensor:
        map_predictions: list[Tensor] = []
        for idx, sub_model in enumerate(self.submodels):
            prediction = sub_model.mlp(geo_tensor)
            if sub_model.cnn_autoencoder is not None:
                assert isinstance(sub_model.cnn_autoencoder, Conv3dAutoencoder)
                prediction = sub_model.cnn_autoencoder.decode(prediction)
            map_predictions.append(prediction)

        return torch.cat(map_predictions, dim=-1)


class SurrogateModelTrainer:
    def __init__(self, nn_trainer: NNTrainer) -> None:
        self.nn_trainer: NNTrainer = nn_trainer

    def fit(
        self,
        model: SurrogateModel,
        data: TrainValidationTestDataTensors,
        autoencoders: AutoencoderCollection | None = None,
    ) -> SurrogateModel:

        channel_count: int = len(OUTPUT_NEURON_LIST)

        if len(model.submodels) != channel_count:
            raise ValueError("The surrogate must contain one MLP per output channel.")

        if len({id(part.mlp) for part in model.submodels}) != channel_count:
            raise ValueError("Each output channel requires an independent MLP.")

        for split_name, split in (
            ("train", data.train),
            ("validation", data.validation),
        ):
            if (
                split.map_tensor.ndim != 5
                or split.map_tensor.shape[-1] != channel_count
                or split.map_tensor.shape[0] == 0
                or split.geometry_tensor.ndim != 2
                or split.geometry_tensor.shape[0] != split.map_tensor.shape[0]
            ):
                raise ValueError(f"Invalid surrogate training data: {split_name}")

        if (
            data.train.map_tensor.shape[1:] != data.validation.map_tensor.shape[1:]
            or data.train.geometry_tensor.shape[1]
            != data.validation.geometry_tensor.shape[1]
        ):
            raise ValueError("Training and validation dimensions must match.")

        if any(part.cnn_autoencoder is not None for part in model.submodels):
            raise ValueError(
                "Supply CNNs through AutoencoderCollection. "
                "The input surrogate must contain MLPs only."
            )

        frozen_autoencoders: AutoencoderCollection | None = None

        if autoencoders is not None:
            autoencoders.validate_compatible(data)

            copied_collection: AutoencoderCollection = copy.deepcopy(autoencoders)
            copied_collection.freeze()
            frozen_autoencoders = copied_collection

        map_shape: tuple[int, ...] = tuple(data.train.map_tensor.shape[1:-1])
        input_count: int = data.train.geometry_tensor.shape[1]

        # Validate every MLP before starting the first training run.
        for channel_index, channel_name in enumerate(OUTPUT_NEURON_LIST):
            part = model.submodels[channel_index]

            expected_output: tuple[int, ...] = (*map_shape, 1)

            if frozen_autoencoders is not None:
                expected_output = (
                    frozen_autoencoders.for_channel(channel_name).latent_dim,
                )

            if part.mlp.num_input_neurons != input_count:
                raise ValueError(f"Wrong MLP input size for channel: {channel_name}")

            if part.mlp.output_dims != expected_output:
                raise ValueError(
                    f"Wrong MLP output dimensions for {channel_name}: "
                    f"expected {expected_output}, "
                    f"received {part.mlp.output_dims}."
                )

        for channel_index, channel_name in enumerate(OUTPUT_NEURON_LIST):
            part = model.submodels[channel_index]
            channel_autoencoder: Conv3dAutoencoder | None = None

            if frozen_autoencoders is not None:
                channel_autoencoder = frozen_autoencoders.for_channel(channel_name)

            logger.info(
                "Training surrogate channel %s (%d/%d), frozen CNN: %s",
                channel_name,
                channel_index + 1,
                channel_count,
                channel_autoencoder is not None,
            )

            trained_network: MLP | Conv3dAutoencoder = self.nn_trainer.start(
                neuronal_network=part.mlp,
                train_data=data.train.select_channel(channel_index),
                validation_data=data.validation.select_channel(channel_index),
                trained_autoencoder=channel_autoencoder,
            )

            if not isinstance(trained_network, MLP):
                raise TypeError("Surrogate training must return an MLP.")

            trained_network.eval()
            part.mlp = trained_network
            part.cnn_autoencoder = channel_autoencoder

        logger.info(
            "Completed surrogate training for %d channels",
            channel_count,
        )
        return model
