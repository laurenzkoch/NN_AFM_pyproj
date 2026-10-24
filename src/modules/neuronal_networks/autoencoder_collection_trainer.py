import copy
import logging

from src.config.config_mcad_parameters import (
    AXIS_LIST,
    OUTPUT_NEURON_LIST,
)
from src.modules.neuronal_networks.autoencoder_collection import (
    AutoencoderCollection,
)
from src.modules.neuronal_networks.manipulating_data import (
    TrainValidationTestDataTensors,
)
from src.modules.neuronal_networks.nn_training_torch import NNTrainer
from src.modules.neuronal_networks.NN_types.cnn_autoencoder import (
    Conv3dAutoencoder,
)
from src.modules.neuronal_networks.NN_types.mlp_with_torch import MLP

logger = logging.getLogger(__name__)


class AutoencoderCollectionTrainer:
    def __init__(self, nn_trainer: NNTrainer) -> None:
        self.nn_trainer: NNTrainer = nn_trainer

    def fit(
        self,
        autoencoders: tuple[Conv3dAutoencoder, ...],
        data: TrainValidationTestDataTensors,
    ) -> AutoencoderCollection:

        channel_count: int = len(OUTPUT_NEURON_LIST)

        if len(autoencoders) != channel_count:
            raise ValueError("Exactly one autoencoder per output channel is required.")

        for split_name, split in (
            ("train", data.train),
            ("validation", data.validation),
        ):
            if (
                split.map_tensor.ndim != 5
                or split.map_tensor.shape[-1] != channel_count
                or split.map_tensor.shape[0] == 0
            ):
                raise ValueError(f"Invalid autoencoder training data: {split_name}")

        if data.train.map_tensor.shape[1:] != data.validation.map_tensor.shape[1:]:
            raise ValueError("Training and validation map dimensions must match.")

        collection: AutoencoderCollection = AutoencoderCollection(
            autoencoders=tuple(
                copy.deepcopy(autoencoder) for autoencoder in autoencoders
            ),
            map_axes={
                name: tuple(float(value) for value in data.map_axes[name])
                for name in AXIS_LIST
            },
            target_minimum=data.target_minimum.detach().cpu().clone(),
            target_denominator=(data.target_denominator.detach().cpu().clone()),
            split_machine_ids={
                name: tuple(machine_ids)
                for name, machine_ids in data.split_machine_ids.items()
            },
            source_hashes=dict(data.source_hashes),
        )

        collection.validate_compatible(data)

        expected_shape: tuple[int, ...] = tuple(data.train.map_tensor.shape[1:-1])

        for channel_name, autoencoder in zip(
            OUTPUT_NEURON_LIST,
            collection.autoencoders,
            strict=True,
        ):
            if tuple(autoencoder.map_shape) != expected_shape:
                raise ValueError(
                    f"Autoencoder dimensions differ from the data: {channel_name}"
                )

        for channel_index, channel_name in enumerate(OUTPUT_NEURON_LIST):
            autoencoder: Conv3dAutoencoder = collection.autoencoders[channel_index]
            autoencoder.requires_grad_(True)

            logger.info(
                "Training autoencoder channel %s (%d/%d), features=%s, latent_dim=%d",
                channel_name,
                channel_index + 1,
                channel_count,
                autoencoder.features,
                autoencoder.latent_dim,
            )

            trained_network: MLP | Conv3dAutoencoder = self.nn_trainer.start(
                neuronal_network=autoencoder,
                train_data=data.train.select_channel(channel_index),
                validation_data=data.validation.select_channel(channel_index),
                trained_autoencoder=None,
            )

            if not isinstance(trained_network, Conv3dAutoencoder):
                raise TypeError("Autoencoder training must return a Conv3dAutoencoder.")

            if trained_network is not autoencoder:
                raise RuntimeError("NNTrainer unexpectedly returned a different model.")

        collection.freeze()

        logger.info(
            "Completed autoencoder training for %d channels",
            channel_count,
        )
        return collection
