
import pandas as pd
from matplotlib import pyplot as plt

from V1.modules.utilitis.ml_model_training import MLModel
from V1.modules.utilitis.plotting import Plotter
from V1.config import config_paths
import pickle
from pathlib import Path


def main():

    folder = config_paths.PATH_DATA / "MLP_Research" / "Pickel_files"
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

    colors = ["b", "g", "r", "c", "m", ]
    func = ""
    layer_sizes = ""

    files = sorted(folder.glob(".pkl"))
    n_files = len(files)
    cmap = plt.get_cmap("hsv")
    colors = [cmap(i / max(1, n_files - 1)) for i in range(n_files)]

    for file, color in zip(files, colors):

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
            func = model.function
            layer_sizes = model.layer_sizes


            results = pd.concat(
                [
                    results,
                    pd.DataFrame({
                        "Number of data points": [number_of_data_points],
                        "computing time": [computing_time],
                        "R2": [r2_score],
                        "func": [func],
                        "layer_sizes": [layer_sizes],
                    })
                ],
                ignore_index=True
            )

        plotter.add_plot_to_fig(
            ax=ax1,
            x_column=results["Number of data points"],
            y_column=results["R2"],
            colors=color,
            label=f"{func} {layer_sizes}"
        )
        plotter.add_plot_to_fig(
            ax=ax2,
            x_column=results["Number of data points"],
            y_column=results["computing time"],
            colors=color,
            label=f"{func} {layer_sizes}"
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