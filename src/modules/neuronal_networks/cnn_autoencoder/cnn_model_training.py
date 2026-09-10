import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from modules.neuronal_networks.cnn_autoencoder.cnn_autoencoder import Conv3dAutoencoder

from src.config.config_mcad_parameters import *


class CNNModelTrainer:

    def __init__(
            self,
            num_points_per_dim: int,
            num_machine_designs: int,
            filename: str
    ):
        self.num_points_per_dim = num_points_per_dim
        self.num_machine_designs = num_machine_designs
        self.filename = filename
        self.model = Conv3dAutoencoder(
            latent_dim=16,
            num_output_neurons=len(OUTPUT_NEURON_LIST),
            points_per_dim=num_points_per_dim,
            out_channels=15
        )
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3
        )


    def _load_data(self) -> DataLoader:

        x = create_tensor(
            filename=self.filename,
            num_machine_designs=self.num_machine_designs,
            num_points_per_dim=self.num_points_per_dim
        )
        dataset = TensorDataset(x)

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=True
        )

    def __enter__(self) -> "CNNModelTrainer":
        self.data = self._load_data()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return



    def training(
            self,
            num_epochs: int,
    ) -> Conv3dAutoencoder:

        for epoch in range(num_epochs):

            self.model.train()

            epoch_loss = 0.0

            for (x_batch,) in self.data:

                # 1. Gradienten zurücksetzen
                self.optimizer.zero_grad()

                # 2. Autoencoder
                x_hat = self.model(x_batch)

                # 3. Rekonstruktionsfehler
                loss = self.criterion(
                    x_hat,
                    x_batch
                )

                # 4. Backpropagation
                loss.backward()

                # 5. Gewichte verändern
                self.optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(self.data)

            print(
                f"Epoch {epoch}: "
                f"Loss = {epoch_loss:.6f}"
            )

        return self.model