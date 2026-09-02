import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from modules.cnn_autoencoder.cnn_autoencoder import Conv3dAutoencoder
from modules.cnn_autoencoder.create_tensor import create_tensor

class CNNModelTrainer:

    def __init__(self):

        self.model = Conv3dAutoencoder(
            latent_dim=16,
            num_output_neurons=7,
            points_per_dim=10,
            out_channels=15
        )
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3
        )


    @staticmethod
    def _load_data(filename: str, num_machine_designs: int, num_points_per_dim: int) -> DataLoader:

        x = create_tensor(
            filename=filename,
            num_machine_designs=num_machine_designs,
            num_points_per_dim=num_points_per_dim
        )
        dataset = TensorDataset(x)
        return DataLoader(
            dataset,
            batch_size=30,
            shuffle=True
        )

    def training(
            self,
            filename: str,
            num_epochs: int,
            num_machine_designs:int,
            num_points_per_dim: int
    ) -> Conv3dAutoencoder:

        data = self._load_data(
            filename=filename,
            num_machine_designs=num_machine_designs,
            num_points_per_dim=num_points_per_dim
        )

        for epoch in range(num_epochs):

            self.model.train()

            epoch_loss = 0.0

            for (x_batch,) in data:

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

            epoch_loss /= len(data)

            print(
                f"Epoch {epoch}: "
                f"Loss = {epoch_loss:.6f}"
            )

        return self.model