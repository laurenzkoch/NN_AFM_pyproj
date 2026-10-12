import copy
from dataclasses import dataclass

import torch
from torch import Tensor

from src.config.config_mcad_parameters import OUTPUT_NEURON_LIST
from src.modules.neuronal_networks.NN_types.cnn_autoencoder import Conv3dAutoencoder
from src.modules.neuronal_networks.NN_types.mlp_with_torch import MLP
from src.modules.neuronal_networks.manipulating_data import MachineDataTensor
from src.modules.neuronal_networks.nn_training_torch import NNTrainer


@dataclass
class SubSurrogateModel:
    mlp: MLP
    cnn_autoencoder: Conv3dAutoencoder | None = None


class SurrogateModelTrainer:
    def __init__(self, nn_trainer: NNTrainer):
        self.nn_trainer = nn_trainer

    def fit(
            self,
            mlp: MLP,
            train_data: MachineDataTensor,  # (geo_tensor[G, 13], map_tensor[G, *map_size, 1])
            validation_data: MachineDataTensor,
            cnn_autoencoder: Conv3dAutoencoder | None = None,
    ) -> SurrogateModel:

        submodels: list[SubSurrogateModel] = []
        for idx_channel, _ in enumerate(OUTPUT_NEURON_LIST):

            channel_mlp: MLP = copy.deepcopy(mlp)
            channel_cnn_autoencoder: Conv3dAutoencoder | None = copy.deepcopy(cnn_autoencoder)

            _train_data = MachineDataTensor(
                geometry_tensor=train_data.geometry_tensor,
                map_tensor=train_data.map_tensor[..., idx_channel:idx_channel + 1],
            )
            _validation_data = MachineDataTensor(
                geometry_tensor=validation_data.geometry_tensor,
                map_tensor=validation_data.map_tensor[..., idx_channel:idx_channel + 1],
            )

            if channel_cnn_autoencoder is None:
                trained_mlp = self.nn_trainer.start(
                    neuronal_network=channel_mlp,
                    trained_autoencoder=None,
                    train_data=_train_data,
                    validation_data=_validation_data,
                )
                trained_autoencoder = None
            else:
                trained_autoencoder = self.nn_trainer.start(
                    neuronal_network=channel_cnn_autoencoder,
                    trained_autoencoder=None,
                    train_data=_train_data,
                    validation_data=_validation_data,
                )
                assert isinstance(trained_autoencoder, Conv3dAutoencoder)
                trained_mlp = self.nn_trainer.start(
                    neuronal_network=channel_mlp,
                    trained_autoencoder=trained_autoencoder,
                    train_data=_train_data,
                    validation_data=_validation_data,
                )

            assert isinstance(trained_mlp, MLP)
            submodels.append(
                SubSurrogateModel(
                    mlp=trained_mlp,
                    cnn_autoencoder=trained_autoencoder,
                )
            )
        return SurrogateModel(submodels)


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
