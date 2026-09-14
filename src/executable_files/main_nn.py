import math

import torch
import torch.nn as nn

from src.modules.neuronal_networks.mlp_with_torch import MLP
from src.modules.neuronal_networks.cnn_autoencoder import Conv3dAutoencoder
from src.modules.neuronal_networks.nn_training_torch import NNTrainer

from src.config.config_mcad_parameters import *
from src.config.config_paths import *

file = "Results_EMagnetic_simulation_3_3_5_10.09.2026_14-24-32_CEST.csv"

mlp = MLP(
    num_input_neurons=len(INPUT_NEURON_LIST),
    num_output_neurons=len(OUTPUT_NEURON_LIST)* math.prod((5, 5, 5)),
    hidden_layer_neurons=(64, 32, 16),
    activation=nn.ReLU
)

autoencoder = Conv3dAutoencoder(
    latent_dim=15,
    features=(len(OUTPUT_NEURON_LIST), 2*len(OUTPUT_NEURON_LIST), 3*len(OUTPUT_NEURON_LIST)),
    map_shape=(5, 5, 5),
    kernel_size=3,
    activation=nn.ReLU,
)


mlp_trainer = NNTrainer(
    neuronal_network=mlp,
    data_filename=file,
    learning_rate=0.001,
    weight_decay=0.0001,
    num_epochs=1000,
    patience=50,
)
trained_mlp = mlp_trainer.training_process()


autoencoder_trainer = NNTrainer(
    neuronal_network=autoencoder,
    data_filename=file,
    learning_rate=0.001,
    weight_decay=0.0001,
    num_epochs=1000,
    patience=50,
)
trained_autoencoder = autoencoder_trainer.training_process()
model_path = PATH_DATA / "trained_autoencoder"
model_path.mkdir(
    parents=True,
    exist_ok=True
)
torch.save(
    trained_autoencoder.state_dict(),
    model_path / "trained_autoencoder.pt"
)



mlp = MLP(
    num_input_neurons=len(INPUT_NEURON_LIST),
    num_output_neurons=15,
    hidden_layer_neurons=(64, 32, 16),
    activation=nn.ReLU
)

mlp_cnn_trainer = NNTrainer(
    neuronal_network=mlp,
    trained_autoencoder=trained_autoencoder if not isinstance(trained_autoencoder, MLP) else None,
    data_filename=file,
    learning_rate=0.001,
    weight_decay=0.0001,
    num_epochs=1000,
    patience=50,
)
trained_mlp_cnn = mlp_cnn_trainer.training_process()