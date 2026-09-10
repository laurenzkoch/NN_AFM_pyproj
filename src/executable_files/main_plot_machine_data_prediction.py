import numpy as np

from modules.neuronal_networks.mlp_model import MLPModel, MLPModelTrainer
from Archiv.src.data_managing import *
from modules.utilitis.plotting import Plotter
from config import config_paths
from config.config_mcad_parameters import OUTPUT_NEURON_LIST, INPUT_NEURON_LIST


def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:

    folder= config_paths.PATH_DATA_MCAD_RESULTS
    files = sorted(folder.glob("*.csv"))

    data_df = pd.DataFrame(
        columns=INPUT_NEURON_LIST + OUTPUT_NEURON_LIST
    )

    for file in files:
        data: pd.DataFrame = import_data_from_csv(
            path_data=config_paths.PATH_DATA_MCAD_RESULTS,
            filename=file.name
        )
        data_df = pd.concat(
            [data_df, data],
            ignore_index=True
        )

    x, y = split_data(
        data_df=data_df,
        number_of_outputs=len(OUTPUT_NEURON_LIST)
    )

    return x, y

def plot_prediction(trained_model: MLPModel):

    plotter = Plotter()

    for z_parameter, unit in zip(OUTPUT_NEURON_LIST, ["[Nm]", "[%]", "[W]", "[W]", "[W]", "[W]"]):

        fig, ax = plotter.create_fig_matplotlib(
            xlabel=f"Actual {z_parameter} {unit}",
            ylabel=f"Predicted {z_parameter} {unit}",
            title="Prediction vs MCAD Simulation",
        )

        ax = plotter.add_scatter_to_fig(
            ax=ax,
            x_column=trained_model.y[z_parameter],
            y_column=trained_model.y_pred[z_parameter],
            label="Data Points",
        )


        start = trained_model.y[z_parameter].min() - 10
        end = trained_model.y[z_parameter].max() + 10
        ax = plotter.add_plot_to_fig(
            ax=ax,
            x_column=np.linspace(start, end, 100),
            y_column=np.linspace(start, end, 100),
            colors="b",
            label="ideal",
        )

        ax = plotter.add_legend_to_fig(ax=ax)
        if trained_model.R2_pred is not None:
            plotter.add_text_to_fig(
                ax,
                text=f"R2: {trained_model.R2_pred:.4f}\n"
                     f"Number of training data points: {trained_model.x_train.shape[0]}\n"
                     f"Split: {trained_model.split}\n"
                     f"Solver: {trained_model.solver}\n"
                     f"Activation: {trained_model.activation}\n"
                     f"Learning Rate: {trained_model.learning_rate_init}\n"
                     f"Computing time: {trained_model.computing_time:.2f}s\n",
            )

        plotter.save_plot(
            fig,
            path_plots=config_paths.PATH_DATA_PLOTS / "main_investigation",
            filename=f"{z_parameter}_prediction_vs_sim_{trained_model.split}_norm.png"
        )

def plot_r2(models: list[MLPModel]):

    plotter = Plotter()

    fig, ax = plotter.create_fig_matplotlib(
        xlabel=f"Training Points",
        ylabel=f"R2",
        title="Prediction vs MCAD Simulation",
    )

    plotter.add_plot_to_fig(
        ax=ax,
        x_column=[models[i].split * len(models[i].x_train) for i in range(len(models))],
        y_column=[models[i].R2_pred for i in range(len(models))],
        label="R2",
    )
    plotter.add_legend_to_fig(ax=ax)
    plotter.save_plot(
        fig,
        path_plots=config_paths.PATH_DATA_PLOTS / "main_investigation",
        filename=f"R2_norm_dp.png"
    )

def main():

    trainer = MLPModelTrainer()
    x, y = get_data()
    data_name = "all datasets"

    models: list[MLPModel] = []

    for split in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

        trained_model: MLPModel = trainer.train_model(
            data_name=data_name,
            x=x,
            y=y,
            test_split=split,
            model=trainer.create_model()
        )

        plot_prediction(trained_model=trained_model)
        models.append(trained_model)

        print(f"Split: {trained_model.split}")

    plot_r2(models)



if __name__ == "__main__":
    main()


