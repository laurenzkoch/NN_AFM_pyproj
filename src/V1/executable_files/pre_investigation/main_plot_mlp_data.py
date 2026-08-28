
from V1.modules.utilitis.ml_model_training import MLModel

from V1.config import config_paths
from V1.modules.utilitis.plotting import Plotter
from pathlib import Path
import pickle



def main():

    folder = config_paths.PATH_DATA_MLP_RESEARCH_BASIC
    files = sorted(folder.glob("*.pkl"), key=lambda x: int(x.stem.split('_')[-1]))

    models_data_quantity_sin_exp: list[MLModel] = []
    models_data_quantity_lin: list[MLModel] = []
    models_data_neuron_variation_exp: list[MLModel] = []
    models_data_neuron_variation_lin: list[MLModel] = []

    for file in files:

        with open(Path(config_paths.PATH_DATA / file), "rb") as f:
            model_data: MLModel = pickle.load(f)
            if model_data.data_name == "sin_exp":
                if model_data.layer_sizes == (100, 100, 100):
                    models_data_quantity_sin_exp.append(model_data)
                    #models_data_neuron_variation_exp.append(model_data)
                else:
                    models_data_neuron_variation_exp.append(model_data)
            elif model_data.data_name == "linear":
                if model_data.layer_sizes == (100, 100, 100):
                    models_data_quantity_lin.append(model_data)
                    #models_data_neuron_variation_lin.append(model_data)
                else:
                    models_data_neuron_variation_lin.append(model_data)

    print(f"Number of models qua sin: {len(models_data_quantity_sin_exp)}")
    print(f"Number of models qua lin: {len(models_data_quantity_lin)}")
    print(f"Number of models neu sin: {len(models_data_neuron_variation_exp)}")
    print(f"Number of models neu lin: {len(models_data_neuron_variation_lin)}")

    model_data_df = {
            "quantity_sin_exp": models_data_quantity_sin_exp,
            "quantity_lin": models_data_quantity_lin,
            "neuron_variation_exp": models_data_neuron_variation_exp,
            "neuron_variation_lin": models_data_neuron_variation_lin,
        }

    plotter = Plotter()
    fig2, ax2 = plotter.create_fig_matplotlib(
        xlabel="Number of absolut data points",
        ylabel="computing time [s]",
        title="computing time vs Number of data points"
    )
    for (key, value_list), color1 in zip(model_data_df.items(), ["b", "g", "r", "c"]):

        func = ""
        if key == "quantity_sin_exp" or key == "neuron_variation_exp":
            func = "y = 100 * sin(0.8 * t) * exp(-t/50)"
        elif key == "quantity_lin" or key == "neuron_variation_lin":
            func = "y = 1/50 * t + 10"


        if key != "neuron_variation_exp" and key != "neuron_variation_lin":
            fig1, ax1 = plotter.create_fig_matplotlib(
                xlabel="Number of Data Points",
                ylabel="R^2",
                title="R2 Score vs Number of data points"
            )
            fig4, ax4 = plotter.create_fig_matplotlib(
                xlabel="Number of Data Points",
                ylabel="MSE",
                title="MSE vs Number of data points"
            )
            for r2_name, color in zip(["R2_pred", "R2_train_pred", "R2_test_pred"], ["b", "g", "r"]):
                plotter.add_plot_to_fig(
                    ax=ax1,
                    x_column=[len(value_list[i * 10].x) for i in range(len(value_list) // 10)],
                    y_column=[getattr(value_list[i * 10], r2_name) for i in range(len(value_list) // 10)],
                    colors=color,
                    label=f"{r2_name}"
                )
            for mse_name, color in zip(["mse", "mse_train", "mse_test"], ["b", "g", "r"]):
                plotter.add_plot_to_fig(
                    ax=ax4,
                    x_column=[len(value_list[i * 10].x) for i in range(len(value_list) // 10)],
                    y_column=[getattr(value_list[i * 10], mse_name) for i in range(len(value_list) // 10)],
                    colors=color,
                    label=f"{mse_name}"
                )
            ax1 = plotter.add_text_to_fig(
                        ax=ax1,
                        text=f"Layer Sizes: {value_list[0].layer_sizes}\n"
                             f"Split: 0.20\n"
                             f"Function: {func}"
                    )
            ax4 = plotter.add_text_to_fig(
                        ax=ax4,
                        text=f"Layer Sizes: {value_list[0].layer_sizes}\n"
                             f"Split: 0.20\n"
                             f"Function: {func}"
                    )
            plotter.add_legend_to_fig(ax1)
            plotter.save_plot(fig1, path_plots=Path("MLP_research"), filename=f"result_plot_r2_over_number_of_data_points_{key}.png")
            plotter.add_legend_to_fig(ax4)
            plotter.save_plot(fig4, path_plots=Path("MLP_research"), filename=f"result_plot_mse_over_number_of_data_points_{key}.png")

            plotter.add_plot_to_fig(
                ax=ax2,
                x_column=[len(value_list[i * 10].x) for i in range(len(value_list) // 10)],
                y_column=[getattr(value_list[i * 10], "computing_time") for i in range(len(value_list) // 10)],
                label=f"computing time {func}",
                colors=color1
            )
            ax2 = plotter.add_text_to_fig(
                        ax=ax2,
                        text=f"Layer Sizes: {value_list[0].layer_sizes}\n"
                             f"Split: 0.20"
                    )


        if key == "neuron_variation_exp" or key == "neuron_variation_lin":

            for r2_name, color in zip(["R2_pred", "R2_train_pred", "R2_test_pred"], ["b", "g", "r"]):
                fig3, ax3 = plotter.create_fig_matplotlib(
                    xlabel="Number of Data Points",
                    ylabel="R^2",
                    title="Neuron Variation"
                )
                plotter.add_heatmap_to_fig(
                    ax=ax3,
                    x=[sum(value_list[i].layer_sizes) / len(value_list[i].layer_sizes) for i in range(len(value_list))],
                    y=[len(value_list[i].layer_sizes) for i in range(len(value_list))],
                    z=[getattr(value_list[i], r2_name) for i in range(len(value_list))],
                    label_x="Mean Number of Neurons per Layer",
                    label_y="Number of Layers",
                    label_z=f"{r2_name}",
                    title="R2 Score vs Number of Layers and Mean Number of Neurons per Layer"
                )
                plotter.add_text_to_fig(
                    ax=ax3,
                    text=f"Number of training data points {len(value_list[0].x_train)}\n"
                         f"Split: 0.20\n"
                         f"Function: {func}",
                        x=1.25,
                        y=0.95
                )
                plotter.save_plot(fig3, path_plots=Path("MLP_research"),
                                  filename=f"result_plot_{r2_name}_{key}.png")

    plotter.add_legend_to_fig(ax2)
    plotter.save_plot(fig2, path_plots=Path("MLP_Research_basic_data"), filename=f"result_plot_computing_time.png")

if __name__ == "__main__":
    main()
