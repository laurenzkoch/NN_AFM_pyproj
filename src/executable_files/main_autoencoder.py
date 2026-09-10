
import torch
from modules.neuronal_networks.cnn_autoencoder import CNNModelTrainer
from config.config_paths import PATH_DATA

with CNNModelTrainer(
        num_points_per_dim=5,
        num_machine_designs=5,
        filename="Results_EMagnetic_simulation_1_1_5_5_09.09.2026_06-01-43_CEST.csv"
) as trainer:

    autoencoder = trainer.training(num_epochs=1000)

    model_path = PATH_DATA / "autoencoder"
    model_path.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        autoencoder.state_dict(),
        model_path / "autoencoder.pt"
    )


