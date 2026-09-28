
import torch
import numpy as np

from src.modules.plotting.matplotlib_plotting import *
from src.modules.neuronal_networks.cnn_autoencoder import Conv3dAutoencoder
from src.modules.neuronal_networks.mlp_with_torch import MLP
from src.config.config_mcad_parameters import *
from src.config.config_paths import *


def plot_pred_vs_sim(
        trained_nn: MLP | Conv3dAutoencoder,
        map_std: torch.Tensor,
        map_mean: torch.Tensor,
        nn_inputs: list[torch.Tensor],
        map_sim_list_norm: list[torch.Tensor],
        ref_op_idx: tuple,
        ref_op: tuple
) -> None:

    map_pred_list = []
    for nn_input in nn_inputs:
        map_pred = trained_nn(nn_input) * map_std + map_mean
        map_pred_list.append(map_pred)

    map_sim_list = []
    for map_sim in map_sim_list_norm:
        map_sim = map_sim * map_std + map_mean
        map_sim_list.append(map_sim)

    for idx_channel, (parameter, unit) in enumerate(
            zip(OUTPUT_NEURON_LIST, ["[Nm]", "[%]", "[W]", "[W]", "[W]", "[W]", "[W]"])
    ):

        fig, ax = create_fig_matplotlib(
            xlabel=f"Actual {parameter} {unit}",
            ylabel=f"Predicted {parameter} {unit}",
            title="Prediction vs MCAD Simulation",

        )

        datasets_points = []
        rs = []
        r2 = []
        diff = []
        for map_sim, map_pred in zip(map_sim_list, map_pred_list):
            sim_points = (
                map_sim[:,:,:,:, idx_channel] # all points: map_sim[:,:,:,:, idx_channel]
                .detach()
                .cpu()
                .numpy()
            )

            pred_points = (
                map_pred[:,:,:,:, idx_channel]
                .detach()
                .cpu()
                .numpy()
            )

            datasets_points.append(
                (sim_points, pred_points)
            )

            diff.append(torch.abs(map_sim[:,:,:,:, idx_channel] - map_pred[:,:,:,:, idx_channel]))
            ss_res = torch.sum((map_sim[:,:,:,:, idx_channel] - map_pred[:,:,:,:, idx_channel]) ** 2)
            ss_tot = torch.sum((map_sim[:,:,:,:, idx_channel] - torch.mean(map_sim[:,:,:,:, idx_channel])) ** 2)
            r2.append(1 - ss_res / ss_tot)

            rs.append(
                torch.corrcoef(
                    torch.stack((map_sim[:,:,:,:, idx_channel].flatten(), map_pred[:,:,:,:, idx_channel].flatten()))
                )[0, 1]
            )

        all_points = np.concatenate(
            [
                points
                for dataset in datasets_points
                for points in dataset
            ]
        )

        start = all_points.min() - 0.1
        end = all_points.max() + 0.1
        add_plot_to_fig(
            ax=ax,
            x_column=np.linspace(start, end, 100),
            y_column=np.linspace(start, end, 100),
            colors="b",
            label="ideal",
        )

        colors = ["r", "g"]
        labels = [
            f"Trained machine designs: {len(datasets_points[0][0])}",
            f"Not seen in training: {len(datasets_points[1][0])}"
        ]

        for (sim_points, pred_points), col, label in zip(
                datasets_points,
                colors,
                labels,
        ):
            add_scatter_to_fig(
                ax=ax,
                x_column=sim_points,
                y_column=pred_points,
                label=label,
                colors=col
            )

        add_legend_to_fig(ax=ax)
        add_text_to_fig(
            ax=ax,
            text=f"R train = {rs[0]:.3f}\n"
                 f"R validation = {rs[1]:.3f}\n"
                 f"R^2 train = {r2[0]:.3f}\n"
                 f"R^2 validation = {r2[1]:.3f}\n"
                 f"Min. Error train = {diff[0].min().item():.3f} {unit}\n"
                 f"Max. Error train = {diff[0].max().item():.3f} {unit}\n"
                 f"Min. Error validation = {diff[1].min().item():.3f} {unit}\n"
                 f"Max. Error validation = {diff[1].max().item():.3f} {unit}\n"
                 #f"Operation Point\n"
                 #f"i_d = {ref_op[0]}\n"
                 #f"i_q =  {ref_op[1]}\n"
                 #f"n =  {ref_op[2]}"
        )

        save_plot(
            fig,
            path_plots=PATH_DATA_PLOTS,
            filename=f"{parameter}_prediction_vs_sim_{trained_nn.__class__.__name__}.png"
        )


    print("##### Plots Created #####")











