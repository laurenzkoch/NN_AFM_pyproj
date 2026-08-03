import numpy as np
import pandas as pd

from modules.utilitis.ml_model_training import MLModel, MLModelTrainer
from modules.utilitis.data_managing import DataManager
from modules.utilitis.plotting import Plotter
from src.config import config_paths



def main():

    datamanager = DataManager()
    trainer = MLModelTrainer()

    data: pd.DataFrame = datamanager.import_data_from_csv(
        path_data=config_paths.PATH_DATA_MCAD_RESULTS / "results",
        filename="Results_EMagnetic_simulation_PeakCurrent_PhaseAdvance_17.07.2026_16-29-43_CEST.csv"
    )
    data = data[["PeakCurrent", "PhaseAdvance", "ShaftTorque", "SystemEfficiency", "RotorIronLoss_Total"]]


    x, y = datamanager.split_data(
        data_df=data,
        number_of_outputs=3
    )
    trained_model: MLModel = trainer.train_model(
        data_name="test",
        x=x,
        y=y,
        test_split=0.2,
        model=trainer.create_model()
    )

    plotter = Plotter()
    fig, ax = plotter.create_fig_matplotlib(
        xlabel="Peak Current [A]",
        ylabel="Phase Advance [Degrees]",
        title="EMagnetic Simulation",
    )
    ax = plotter.add_scatter_to_fig(
        ax=ax,
        x_column=trained_model.x_train["PhaseAdvance"],
        y_column=trained_model.x_train["PeakCurrent"],
        colors="b",
        marker="x",
        label="trainig data"
    )
    ax = plotter.add_scatter_to_fig(
        ax=ax,
        x_column=trained_model.x_test["PhaseAdvance"],
        y_column=trained_model.x_test["PeakCurrent"],
        colors="r",
        marker="x",
        label="test data"
    )
    plotter.add_legend_to_fig(ax=ax)
    plotter.save_plot(
        fig,
        path_plots=config_paths.PATH_DATA_PLOTS / "MLP_Research_machine_data",
        filename="EMagnetic_prediction.png"
    )


    for z_parameter, unit in zip(["ShaftTorque", "SystemEfficiency", "RotorIronLoss_Total"],["[Nm]", "[%]", "[W]"]):


        fig, ax = plotter.create_fig_matplotlib(
            xlabel="Peak Current [A]",
            ylabel="Phase Advance [Degrees]",
            title="Prediction",
        )
        ax = plotter.add_heatmap_to_fig(
            ax=ax,
            x=trained_model.x["PhaseAdvance"],
            y=trained_model.x["PeakCurrent"],
            z=trained_model.y_pred[z_parameter],
            label_x="Phase Advance [Degrees]",
            label_y="Peak Current [A]",
            label_z=f"{z_parameter} {unit}",
            title="MCAD Simulation",
            #log_scale=True if z_parameter == "SystemEfficiency" else False,
        )
        plotter.save_plot(
            fig,
            path_plots=config_paths.PATH_DATA_PLOTS / "MLP_Research_machine_data",
            filename=f"{z_parameter}_prediction.png" if z_parameter != "SystemEfficiency" else f"{z_parameter}_prediction.png"
        )


        fig1, ax1 = plotter.create_fig_matplotlib(
            xlabel="Peak Current [A]",
            ylabel="Phase Advance [Degrees]",
            title="Prediction",
        )
        diff = (trained_model.y_pred[z_parameter].reset_index(drop=True) - trained_model.y[z_parameter].reset_index(
            drop=True))
        ax1 = plotter.add_heatmap_to_fig(
            ax=ax1,
            x=trained_model.x["PhaseAdvance"],
            y=trained_model.x["PeakCurrent"],
            z=diff,
            label_x="Phase Advance [Degrees]",
            label_y="Peak Current [A]",
            label_z=f"Difference true and predicted data: {z_parameter} {unit}",
            title="MCAD Simulation",
            log_scale=False
        )
        plotter.save_plot(
            fig1,
            path_plots=config_paths.PATH_DATA_PLOTS / "MLP_Research_machine_data",
            filename=f"{z_parameter}_prediction_diff.png"
        )

        fig2, ax2 = plotter.create_fig_matplotlib(
            xlabel=f"Actual {z_parameter} {unit}",
            ylabel=f"Predicted {z_parameter} {unit}",
            title="Prediction vs MCAD Simulation",
        )

        ax2 = plotter.add_scatter_to_fig(
            ax=ax2,
            x_column=y[z_parameter],
            y_column=trained_model.y_pred[z_parameter],
        )

        match z_parameter:
            case "SystemEfficiency":
                start = -10
                end = 6000
            case "RotorIronLoss_Total":
                start = -10
                end = 850
            case "ShaftTorque":
                start = -350
                end = 350
            case _:
                start = -10
                end = -10


        ax2 = plotter.add_plot_to_fig(
            ax=ax2,
            x_column=np.linspace(start, end, 1024),
            y_column=np.linspace(start, end, 1024),
            colors="b",
        )
        plotter.save_plot(
            fig2,
            path_plots=config_paths.PATH_DATA_PLOTS / "MLP_Research_machine_data",
            filename=f"{z_parameter}_prediction_vs_sim.png"
    )


if __name__ == "__main__":
    main()