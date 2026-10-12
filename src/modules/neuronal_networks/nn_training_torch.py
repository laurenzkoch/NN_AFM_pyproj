import copy
from typing import Any

import torch

from src.modules.neuronal_networks.NN_types.cnn_autoencoder import Conv3dAutoencoder
from src.modules.neuronal_networks.NN_types.mlp_with_torch import MLP
from src.modules.neuronal_networks.manipulating_data import MachineDataTensor


class NNTrainer:

    def __init__(
            self,
            learning_rate: float = 1e-3,  # not smaller than 0.00001
            weight_decay: float = 1e-3,  #
            num_epochs: int = 2000,
            patience: int = 50,
    ) -> None:

        if patience < 1:
            raise ValueError("patience must be >= 1")
        if num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")

        # device
        self.torch_device = torch.device("cpu")

        # loss function
        self.criterion = torch.nn.MSELoss()

        # trainer settings
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.patience = patience

        # data
        self.train_data: MachineDataTensor
        self.validation_data: MachineDataTensor

        self.best_validation_loss = float("inf")
        self.best_model_state = None
        self.epochs_without_improvement: int
        self.optimizer = None

        self.neuronal_network: MLP | Conv3dAutoencoder
        self.trained_autoencoder: Conv3dAutoencoder | None

    def start(
            self,
            neuronal_network: MLP | Conv3dAutoencoder,
            train_data: MachineDataTensor,  # (geo_tensor[G, 13], map_tensor[G, *map_size, 1])
            validation_data: MachineDataTensor,
            trained_autoencoder: Conv3dAutoencoder | None = None,
    ) -> MLP | Conv3dAutoencoder:

        if isinstance(neuronal_network, Conv3dAutoencoder) and trained_autoencoder is not None:
            raise ValueError("trained_autoencoder cannot be used with Conv3dAutoencoder")

        self.neuronal_network = neuronal_network
        self._move_to_device(self.neuronal_network)

        # trained_autoencoder
        self.trained_autoencoder = trained_autoencoder
        self._move_to_device(self.trained_autoencoder)
        self._freeze_autoencoder()

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.neuronal_network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # data
        self.train_data = train_data
        self.train_data.shuffle()
        self.validation_data = validation_data

        self.best_validation_loss = float("inf")
        self.best_model_state = None
        self.epochs_without_improvement = 0

        return self._training_process()

    def _move_to_device(self, an_object: Any) -> Any:
        if an_object is not None:
            return an_object.to(self.torch_device)
        return None

    def _freeze_autoencoder(self) -> None:
        if self.trained_autoencoder is not None:
            self.trained_autoencoder.eval()  # switch to evalutation mode
            for parameter in self.trained_autoencoder.parameters():
                parameter.requires_grad = False

    def _get_small_map(self, map_prediction: torch.Tensor) -> torch.Tensor:

        if map_prediction.ndim != 5:
            raise ValueError(
                f"Expected 5D tensor, got shape {map_prediction.shape}"
            )
        map_shape = tuple(map_prediction.shape[1:-1])

        indices = []
        for axis_size in map_shape:
            middle_index = (axis_size - 1) // 2
            indices.append([0, middle_index, axis_size - 1])

        sparse_indices = torch.tensor(
            indices,
            dtype=torch.long,
            device=self.torch_device
        )

        small_map_pred = (
            map_prediction
            .index_select(1, sparse_indices[0])
            .index_select(2, sparse_indices[1])
            .index_select(3, sparse_indices[2])
        )

        return small_map_pred

    def _calc_loss(
            self,
            true_for_training: bool
    ) -> float:

        if true_for_training:
            data = self.train_data
        else:
            data = self.validation_data

        if true_for_training:
            self.optimizer.zero_grad()

        if isinstance(self.neuronal_network, MLP):
            mlp_prediction = self.neuronal_network(data.geometry_tensor)
            if self.trained_autoencoder is not None:
                map_prediction = self.trained_autoencoder.decode(
                    mlp_prediction
                )
                small_map_pred = self._get_small_map(map_prediction)
                small_map = self._get_small_map(data.map_tensor)
                loss = self.criterion(small_map, small_map_pred)
            else:
                map_prediction = mlp_prediction
                loss = self.criterion(data.map_tensor, map_prediction)

        elif isinstance(self.neuronal_network, Conv3dAutoencoder) and self.trained_autoencoder is None:
            map_prediction = self.neuronal_network(data.map_tensor)
            loss = self.criterion(data.map_tensor, map_prediction)
        else:
            raise RuntimeError("Unknown neuronal network type")

        if true_for_training:
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def _train(self) -> None:
        self.neuronal_network.train()
        self._calc_loss(true_for_training=True)

    @torch.no_grad()
    def _validation(self) -> float:
        self.neuronal_network.eval()
        return self._calc_loss(true_for_training=False)

    def _early_stopping(self, validation_loss: float) -> bool:

        if validation_loss < self.best_validation_loss - 1e-6:
            self.best_validation_loss = validation_loss
            self.best_model_state = copy.deepcopy(
                self.neuronal_network.state_dict()
            )
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            return True
        else:
            return False

    def _training_process(self) -> MLP | Conv3dAutoencoder:

        for epoch in range(self.num_epochs):
            self._train()
            validation_loss = self._validation()
            if self._early_stopping(validation_loss):
                print(f"Early stopping after {epoch + 1} epochs.")
                break
        if self.best_model_state is not None:
            self.neuronal_network.load_state_dict(
                self.best_model_state
            )

        return self.neuronal_network
