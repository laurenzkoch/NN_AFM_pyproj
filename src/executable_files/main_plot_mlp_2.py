
import pandas as pd

from modules.ml_model_trainig import MLModel
from src.modules.plotting import Plotter
from config import config_paths
import pickle
from pathlib import Path


def main():

    folder = config_paths.PATH_DATA / "MLP_Research" / "Quantity of training data"
    plotter = Plotter()

    fig1, ax1 = plotter.create_fig_matplotlib(
        xlabel="Number of Data Points",
        ylabel="R^2",
        title="R2 Score vs Number of data points"
    )
    fig2, ax2 = plotter.create_fig_matplotlib(
        xlabel="Number of Data Points",
        ylabel="computing time [s]",
        title="computing time vs Number of data points"
    )
    fig3, ax3 = plotter.create_fig_matplotlib(
        xlabel="Number of Data Points",
        ylabel="computing time [s]",
        title="computing time vs Number of data points"
    )
    ax3_1 = ax3.twinx()

    colors = ["b", "g", "r", "c"]

    for file, color in zip(folder.glob("*.pkl"), colors):

        with open(Path(config_paths.PATH_DATA / file), "rb") as f:
            model_data: list[MLModel] = pickle.load(f)

        results = pd.DataFrame(
            columns = [
                "Number of data points",
                "computing time",
                "R2"
            ]
        )

        for model in model_data:
            number_of_data_points = len(model.x_test) / 0.2 * 0.8
            computing_time = model.computing_time
            r2_score = model.R2_Score

            results = pd.concat(
                [
                    results,
                    pd.DataFrame({
                        "Number of data points": [number_of_data_points],
                        "computing time": [computing_time],
                        "R2": [r2_score],
                    })
                ],
                ignore_index=True
            )

        plotter.add_plot_to_fig(
            ax=ax1,
            x_column=results["Number of data points"],
            y_column=results["R2"],
            colors=color,
            label=file.stem
        )
        plotter.add_plot_to_fig(
            ax=ax2,
            x_column=results["Number of data points"],
            y_column=results["computing time"],
            colors=color,
            label=file.stem
        )
        if file.stem == "model_data_20260714_212600":
            ax3 =  plotter.add_plot_to_fig(
                ax=ax3,
                x_column=results["Number of data points"],
                y_column=results["computing time"],
                colors=color,
                label="computing time"
            )

            ax3_1 = plotter.add_plot_to_fig(
                ax=ax3_1,
                x_column=results["Number of data points"],
                y_column=results["R2"],
                label="R2",
                colors="r"
            )

    handles1, labels1 = ax3.get_legend_handles_labels()
    handles2, labels2 = ax3_1.get_legend_handles_labels()

    plotter.add_legend_to_fig(ax1)
    plotter.add_legend_to_fig(ax2)
    ax3.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper left",
        bbox_to_anchor=(1.1, 1.0),
        borderaxespad=0.0
    )
    plotter.save_plot(fig1, path_plots=Path("MLP_quantity_of_datapoints") / "results", filename=f"result_plot_r2.png")
    plotter.save_plot(fig2, path_plots=Path("MLP_quantity_of_datapoints") / "results", filename=f"result_plot_computing_time.png")
    plotter.save_plot(fig3, path_plots=Path("MLP_quantity_of_datapoints") / "results", filename=f"result_plot_computing_time_r2.png")


if __name__ == "__main__":
    main()