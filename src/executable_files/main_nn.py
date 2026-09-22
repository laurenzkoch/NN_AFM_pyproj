import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from src.modules.neuronal_networks.manipulating_data import DataTensor
from src.modules.neuronal_networks.mlp_with_torch import MLP, TrainedMLP
from src.modules.neuronal_networks.cnn_autoencoder import Conv3dAutoencoder, TrainedCNNAutoencoder
from src.modules.neuronal_networks.nn_training_torch import NNTrainer

from src.config.config_mcad_parameters import *
from src.config.config_paths import *


########################################################################################################################
# load data

file = "Results_EMagnetic_simulation_3_3_5_18.09.2026_13-52-26_CEST.csv"
data = DataTensor(filename=file)

with open(PATH_DATA_TRAINED_NN / "data.pkl", "wb") as file:
    pickle.dump(data, file)

dataset_train_data = TensorDataset(
    data.geo_train_norm,
    data.op_train_norm,
    data.map_train_norm,
)

dataset_validation_data = TensorDataset(
    data.geo_validation_norm,
    data.op_validation_norm,
    data.map_validation_norm,
)

########################################################################################################################
########################################################################################################################
# MLP ONLY

mlp = MLP(
    num_input_neurons=len(INPUT_NEURON_LIST) + 3,
    num_output_neurons=len(OUTPUT_NEURON_LIST),
    hidden_layer_neurons=(64, 32, 16),
    activation=nn.ReLU
)
mlp_trainer = NNTrainer(
    neuronal_network=mlp,
    train_data=dataset_train_data,
    validation_data=dataset_validation_data,
    learning_rate=0.001,
    weight_decay=0.0001,
    num_epochs=1000,
    patience=50,
)
trained_mlp = mlp_trainer.training_process()

torch.save(
    TrainedMLP(
        model_state_dict=trained_mlp.state_dict(),
        num_input_neurons=len(INPUT_NEURON_LIST) + 3,
        num_output_neurons=len(OUTPUT_NEURON_LIST),
        hidden_layer_neurons=(64, 32, 16),
        activation=nn.ReLU
    ),
    PATH_DATA_TRAINED_NN / "trained_mlp.pt"
)



########################################################################################################################
########################################################################################################################
# AUTOENCODER

autoencoder = Conv3dAutoencoder(
    latent_dim=15,
    features=(len(OUTPUT_NEURON_LIST), 2*len(OUTPUT_NEURON_LIST), 3*len(OUTPUT_NEURON_LIST)),
    map_shape=(5, 5, 5),
    kernel_size=3,
    activation=nn.ReLU,
)
autoencoder_trainer = NNTrainer(
    neuronal_network=autoencoder,
    train_data=dataset_train_data,
    validation_data=dataset_validation_data,
    learning_rate=0.001,
    weight_decay=0.0001,
    num_epochs=1000,
    patience=50,
)
trained_autoencoder = autoencoder_trainer.training_process()

torch.save(
    TrainedCNNAutoencoder(
        model_state_dict=trained_autoencoder.state_dict(),
        latent_dim=15,
        features=(len(OUTPUT_NEURON_LIST), 2*len(OUTPUT_NEURON_LIST), 3*len(OUTPUT_NEURON_LIST)),
        map_shape=(5, 5, 5),
        kernel_size=3,
        activation=nn.ReLU,
    ),
    PATH_DATA_TRAINED_NN / "trained_autoencoder.pt"
)


########################################################################################################################
########################################################################################################################
"""

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
    num_input_neurons=len(INPUT_NEURON_LIST) + 3,
    num_output_neurons=15,
    hidden_layer_neurons=(64, 32, 16),
    activation=nn.ReLU
)

mlp_cnn_trainer = NNTrainer(
    neuronal_network=mlp,
    trained_autoencoder=trained_autoencoder
    dataloader_train_data=dataloader_train_data,
    dataloader_validation_data=dataloader_validation_data,
    learning_rate=0.001,
    weight_decay=0.0001,
    num_epochs=1000,
    patience=50,
)
trained_mlp_cnn = mlp_cnn_trainer.training_process()


"""