import copy
from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.modules.neuronal_networks.mlp_model.mlp_with_torch import MLPGeometryToLatent
from src.modules.neuronal_networks.manipulating_data import DataTensor
from src.config.config_mcad_parameters import *






class MLPTrainer:

    def __init__(
            self,
            data_filename: str,
            autoencoder: nn.Module,
            num_output_neurons: int,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            num_epochs: int = 1000,
            patienc: int = 50
    ):
        #device
        self.torch_device = torch.device("cpu")

        #create mlp
        self.mlp = MLPGeometryToLatent(
            num_input_neurons=len(INPUT_NEURON_LIST),
            num_output_neurons=num_output_neurons
        )
        self._move_to_device(self.mlp)

        #autoencoder
        self.autoencoder = autoencoder
        self._move_to_device(self.autoencoder)
        self._freeze_autoencoder()

        # load data
        data = DataTensor(filename=data_filename)
        self.train_data = data.dataloader_train
        self.validation_data = data.dataloader_validation

        #loss function
        self.criterion = nn.MSELoss()

        #optimizer
        self.optimizer = torch.optim.Adam(
            self.mlp.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.num_epochs = num_epochs
        self.validation_loss = 0
        self.patience = patienc
        self.best_model_state = None


    def _move_to_device(self, an_object: Any) -> Any:
        return an_object.to(self.torch_device)


    def _freeze_autoencoder(self) -> None:
        self.autoencoder.eval()  # switch to evalutation mode
        for parameter in self.autoencoder.parameters():
            parameter.requires_grad = False


    def _loss(self, y_tensor_pred: torch.Tensor, y_tensor_train) -> torch.Tensor:
        loss = self.criterion(
            y_tensor_train,
            y_tensor_pred
        )
        return loss


    def _train(self):

        self.mlp.train()
        train_loss = 0.0

        for geometry_batch, map_batch in self.train_data:

            geometry_batch = self._move_to_device(geometry_batch)
            map_batch = self._move_to_device(map_batch)

            self.optimizer.zero_grad()

            z_pred = self.mlp(geometry_batch)

            loss = self._loss(map_batch, z_pred)
            loss.backward()

            self.optimizer.step()

            train_loss += (loss.item() * geometry_batch.size(0))

        train_loss /= len(self.train_data[0])


    def _validation(self):

        self.mlp.eval()
        validation_loss = 0.0

        with torch.no_grad():
            for geometry_batch, map_batch in self.validation_data:

                geometry_batch = self._move_to_device(geometry_batch)
                map_batch = self._move_to_device(map_batch)

                z_pred = self.mlp(
                    geometry_batch
                )

                loss = self._loss(map_batch, z_pred)
                loss.backward()

                validation_loss += (loss.item() * geometry_batch.size(0))

            self.validation_loss /= len(self.validation_data[0])


    def _early_stopping(self) -> bool:

        best_validation_loss = float("inf")
        epochs_without_improvement = 0

        if self.validation_loss < best_validation_loss:

            best_validation_loss = self.validation_loss
            self.best_model_state = copy.deepcopy(
                self.mlp.state_dict()
            )
            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= self.patience:
            return True
        else:
            return False


    def training_process(self):

        for epoch in range(self.num_epochs):
            self._train()
            self._validation()
            if self._early_stopping():
                print(f"Early stopping after {epoch} epochs.")
                break
        if self.best_model_state is not None:
            self.mlp.load_state_dict(
                self.best_model_state
            )







def train_mlp(
        self,
        autoencoder: nn.Module,
        geometry: torch.Tensor,
        small_maps: torch.Tensor,
        num_epochs: int = 1000,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        validation_split: float = 0.2,
        patience: int = 50,
        device: str = "cpu"
) -> nn.Module:

    device = torch.device(device)

    mlp = self.mlp.to(device)
    autoencoder = autoencoder.to(device)

    # --------------------------------------------------
    # 1. Decoder / Autoencoder einfrieren
    # --------------------------------------------------

    autoencoder.eval()

    for parameter in autoencoder.parameters():
        parameter.requires_grad = False

    # --------------------------------------------------
    # 2. Dataset
    # --------------------------------------------------

    dataset = TensorDataset(
        geometry,
        small_maps
    )

    num_validation = int(
        len(dataset) * validation_split
    )

    num_training = (
            len(dataset)
            - num_validation
    )

    train_dataset, validation_dataset = random_split(
        dataset,
        [
            num_training,
            num_validation
        ],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # --------------------------------------------------
    # 3. Loss + Optimizer
    # --------------------------------------------------

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        mlp.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # --------------------------------------------------
    # 4. Indizes kleines -> großes Kennfeld
    #
    # 2 Punkte entsprechen bei 5 Punkten:
    #
    # klein: 0, 1
    # groß : 0, 4
    #
    # --------------------------------------------------

    sparse_indices = torch.tensor(
        [0, 4],
        dtype=torch.long,
        device=device
    )

    # --------------------------------------------------
    # 5. Early Stopping
    # --------------------------------------------------

    best_validation_loss = float("inf")

    best_model_state = None

    epochs_without_improvement = 0

    # --------------------------------------------------
    # 6. Training
    # --------------------------------------------------

    for epoch in range(num_epochs):

        mlp.train()

        train_loss = 0.0

        for (
                geometry_batch,
                small_map_true
        ) in train_loader:
            geometry_batch = geometry_batch.to(device)

            small_map_true = small_map_true.to(device)

            # ------------------------------
            # Gradienten zurücksetzen
            # ------------------------------

            optimizer.zero_grad()

            # ------------------------------
            # Geometrie -> latent
            # ------------------------------

            z_pred = mlp(
                geometry_batch
            )

            # ------------------------------
            # latent -> großes Kennfeld
            #
            # KEIN torch.no_grad()!
            # ------------------------------

            large_map_pred = autoencoder.decode(
                z_pred
            )

            # ------------------------------
            # bekannte FEM-Punkte auswählen
            # ------------------------------

            small_map_pred = (
                large_map_pred
                .index_select(
                    2,
                    sparse_indices
                )
                .index_select(
                    3,
                    sparse_indices
                )
                .index_select(
                    4,
                    sparse_indices
                )
            )

            # ------------------------------
            # Loss
            # ------------------------------

            loss = criterion(
                small_map_pred,
                small_map_true
            )

            # ------------------------------
            # Backpropagation
            # ------------------------------

            loss.backward()

            # ------------------------------
            # nur MLP-Gewichte verändern
            # ------------------------------

            optimizer.step()

            train_loss += (
                    loss.item()
                    * geometry_batch.size(0)
            )

        train_loss /= len(
            train_dataset
        )

        # --------------------------------------------------
        # 7. Validation
        # --------------------------------------------------

        mlp.eval()

        validation_loss = 0.0

        with torch.no_grad():

            for (
                    geometry_batch,
                    small_map_true
            ) in validation_loader:
                geometry_batch = (
                    geometry_batch.to(device)
                )

                small_map_true = (
                    small_map_true.to(device)
                )

                z_pred = mlp(
                    geometry_batch
                )

                large_map_pred = (
                    autoencoder.decode(
                        z_pred
                    )
                )

                small_map_pred = (
                    large_map_pred
                    .index_select(
                        2,
                        sparse_indices
                    )
                    .index_select(
                        3,
                        sparse_indices
                    )
                    .index_select(
                        4,
                        sparse_indices
                    )
                )

                loss = criterion(
                    small_map_pred,
                    small_map_true
                )

                validation_loss += (
                        loss.item()
                        * geometry_batch.size(0)
                )

        validation_loss /= len(
            validation_dataset
        )

        # --------------------------------------------------
        # 8. Ausgabe
        # --------------------------------------------------

        print(
            f"Epoch {epoch + 1:4d} | "
            f"Train Loss: {train_loss:.6e} | "
            f"Validation Loss: "
            f"{validation_loss:.6e}"
        )

        # --------------------------------------------------
        # 9. Early Stopping
        # --------------------------------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = (
                validation_loss
            )

            best_model_state = copy.deepcopy(
                mlp.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping nach "
                f"{epoch + 1} Epochen."
            )

            break

    # --------------------------------------------------
    # 10. bestes MLP wiederherstellen
    # --------------------------------------------------

    if best_model_state is not None:
        mlp.load_state_dict(
            best_model_state
        )

    return mlp