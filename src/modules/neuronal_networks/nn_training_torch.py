import copy
from typing import Any
import torch
import torch.nn as nn

from src.modules.neuronal_networks.mlp_with_torch import MLP
from src.modules.neuronal_networks.cnn_autoencoder import Conv3dAutoencoder
from src.modules.neuronal_networks.manipulating_data import DataTensor


class NNTrainer:

    def __init__(
            self,
            neuronal_network: MLP | Conv3dAutoencoder,
            data_filename: str,
            trained_autoencoder: Conv3dAutoencoder | None = None,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            num_epochs: int = 1000,
            patience: int = 50,
    ) -> None:

        if patience < 1:
            raise ValueError("patience must be >= 1")
        if num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
        if isinstance(neuronal_network, Conv3dAutoencoder) and trained_autoencoder is not None:
            raise ValueError("trained_autoencoder cannot be used with Conv3dAutoencoder")


        #device
        self.torch_device = torch.device("cpu")

        self.neuronal_network = neuronal_network
        self._move_to_device(self.neuronal_network)

        #trained_autoencoder
        self.trained_autoencoder = trained_autoencoder
        self._move_to_device(self.trained_autoencoder)
        self._freeze_autoencoder()

        # load data
        data = DataTensor(filename=data_filename)
        self.train_data = data.dataloader_train
        self.validation_data = data.dataloader_validation

        #loss function
        self.criterion = nn.MSELoss()

        #optimizer
        self.optimizer = torch.optim.Adam(
            self.neuronal_network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.num_epochs = num_epochs
        self.validation_loss = 0
        self.train_loss = 0
        self.patience = patience
        self.best_model_state = None
        self.best_validation_loss = float("inf")
        self.best_train_loss = float("inf")
        self.epochs_without_improvement = 0


    def _move_to_device(self, an_object: Any) -> Any:
        if an_object is not None:
            return an_object.to(self.torch_device)
        return None


    def _freeze_autoencoder(self) -> None:
        if self.trained_autoencoder is not None:
            self.trained_autoencoder.eval()  # switch to evalutation mode
            for parameter in self.trained_autoencoder.parameters():
                parameter.requires_grad = False


    def _autoencoder_prediction(self, small_map_pred: torch.Tensor) -> torch.Tensor:
        if self.trained_autoencoder is not None:
            large_map_pred = self.trained_autoencoder.decode(
                small_map_pred
            )
            if large_map_pred.ndim != 5:
                raise ValueError(
                    f"Expected 5D tensor, got shape {large_map_pred.shape}"
                )
            map_shape = tuple(large_map_pred.shape[2:])

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
                large_map_pred
                .index_select(2, sparse_indices[0])
                .index_select(3, sparse_indices[1])
                .index_select(4, sparse_indices[2])
            )
        return small_map_pred


    def _show_nn_batches(
            self,
            true_for_training: bool
    ) -> float:

        sum_batch_loss = 0.0

        if true_for_training:
            dataloader = self.train_data
        else:
            dataloader = self.validation_data

        for geometry_batch, map_batch in dataloader:
            geometry_batch = self._move_to_device(geometry_batch)
            map_batch = self._move_to_device(map_batch)

            if true_for_training:
                self.optimizer.zero_grad()

            if isinstance(self.neuronal_network, MLP):
                map_pred = self.neuronal_network(geometry_batch)
                map_pred = map_pred.view_as(map_batch)
                if self.trained_autoencoder is not None:
                    map_pred = self._autoencoder_prediction(map_pred)
            elif isinstance(self.neuronal_network, Conv3dAutoencoder) and self.trained_autoencoder is None:
                map_pred = self.neuronal_network(map_batch)
            else:
                raise RuntimeError("Unknown neuronal network type")

            loss = self.criterion(map_batch, map_pred)

            if true_for_training:
                loss.backward()
                self.optimizer.step()

            sum_batch_loss += (loss.item() * map_batch.size(0))

        mean_total_loss = sum_batch_loss / len(dataloader.dataset)

        return mean_total_loss


    def _train(self) -> None:
        self.neuronal_network.train()
        self.train_loss = self._show_nn_batches(true_for_training=True)


    @torch.no_grad()
    def _validation(self) -> None:
        self.neuronal_network.eval()
        self.validation_loss = self._show_nn_batches(true_for_training=False)


    def _early_stopping(self) -> bool:

        if self.validation_loss < self.best_validation_loss - 1e-6:

            self.best_validation_loss = self.validation_loss
            self.best_train_loss = self.train_loss
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


    def training_process(self) -> MLP | Conv3dAutoencoder:

        print("###START TRAINING###")

        for epoch in range(self.num_epochs):
            self._train()
            self._validation()
            if self._early_stopping():
                print(f"Early stopping after {epoch + 1} epochs.")
                break
        if self.best_model_state is not None:
            self.neuronal_network.load_state_dict(
                self.best_model_state
            )
        print(
            f"###END OF TRAINING###\n"
            f"Train loss: {self.best_train_loss}\n"
            f"Validation loss: {self.best_validation_loss}\n"
            f"######################"
        )

        return self.neuronal_network








