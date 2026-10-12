import pickle

import torch
from src.modules.neuronal_networks.create_plots import plot_pred_vs_sim

from modules.neuronal_networks.NN_types.cnn_autoencoder import Conv3dAutoencoder, TrainedCNNAutoencoder
from modules.neuronal_networks.NN_types.mlp_with_torch import MLP, TrainedMLP
from src.config.config_paths import *
from src.modules.neuronal_networks.manipulating_data import DataLoader

########################################################################################################################
# load data

with open(PATH_DATA_TRAINED_NN / "data.pkl", "rb") as file:
    data: DataLoader = pickle.load(file)

ref_op_idx = (2, 2, 2)
ref_op = tuple(
    (
            data.op_train_norm[0, *ref_op_idx, :] * data.op_std.squeeze()
            + data.op_mean.squeeze()
    ).tolist()
)

########################################################################################################################
########################################################################################################################


mlp_config: TrainedMLP = torch.load(
    PATH_DATA_TRAINED_NN / "trained_mlp.pt",
    map_location=torch.device("cpu"),
    weights_only=False,
)

mlp = MLP(num_input_neurons=mlp_config.num_input_neurons, output_dims=mlp_config.num_output_neurons,
          hidden_layer_neurons=mlp_config.hidden_layer_neurons, activation=mlp_config.activation)

mlp.load_state_dict(mlp_config.model_state_dict)
mlp.to(torch.device("cpu"))
mlp.eval()

mlp_input_train_data = torch.cat((data.geo_train_norm, data.op_train_norm,), dim=-1)
mlp_input_validation_data = torch.cat((data.geo_validation_norm, data.op_validation_norm,), dim=-1)

plot_pred_vs_sim(
    trained_nn=mlp,
    map_std=data.map_std,
    map_mean=data.map_mean,
    nn_inputs=[mlp_input_train_data, mlp_input_validation_data],
    map_sim_list_norm=[data.map_train_norm, data.map_validation_norm],
    ref_op_idx=ref_op_idx,
    ref_op=ref_op,
)

cnn_config: TrainedCNNAutoencoder = torch.load(
    PATH_DATA_TRAINED_NN / "trained_autoencoder.pt",
    map_location=torch.device("cpu"),
    weights_only=False,
)

autoencoder = Conv3dAutoencoder(
    latent_dim=cnn_config.latent_dim,
    features=cnn_config.features,
    map_shape=cnn_config.map_shape,
    kernel_size=cnn_config.kernel_size,
    activation=cnn_config.activation,
)

autoencoder.load_state_dict(cnn_config.model_state_dict)
autoencoder.to(torch.device("cpu"))
autoencoder.eval()

plot_pred_vs_sim(
    trained_nn=autoencoder,
    map_std=data.map_std,
    map_mean=data.map_mean,
    nn_inputs=[data.map_train_norm, data.map_validation_norm],
    map_sim_list_norm=[data.map_train_norm, data.map_validation_norm],
    ref_op_idx=ref_op_idx,
    ref_op=ref_op,
)
