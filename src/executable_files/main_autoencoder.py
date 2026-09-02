
import torch
from modules.cnn_autoencoder.cnn_model_training import CNNModelTrainer
from config.config_paths import PATH_DATA

trainer = CNNModelTrainer()

autoencoder = trainer.training(
    filename="Results_EMagnetic_simulation_5_1_1_31.08.2026_17-34-55_CEST.csv",
    num_epochs=100,
    num_points_per_dim=50,
    num_machine_designs=5
)



model_path = PATH_DATA / "autoencoder"
model_path.mkdir(
    parents=True,
    exist_ok=True
)

torch.save(
    autoencoder.state_dict(),
    "autoencoder.pt"
)


