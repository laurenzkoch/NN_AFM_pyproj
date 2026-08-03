
import pandas as pd

from modules.utilitis.ml_model_training import MLModel
from modules.utilitis.plotting import Plotter
from config import config_paths
import pickle
from pathlib import Path


def main():

    ordner = Path(config_paths.PATH_DATA)
    plotter = Plotter()

    for file in ordner.glob("*.pkl"):

        with open(Path(config_paths.PATH_DATA / file), "rb") as f:
            model_data: list[MLModel] = pickle.load(f)

        results = pd.DataFrame(
            columns = [
                "num_of_layers",
                "mean_num_of_neurons_per_layer",
                "R2"
            ]
        )

        for model in model_data:
            num_of_layer = len(model.layer_sizes)
            mean_num_of_neuron_per_layer = sum(model.layer_sizes) / num_of_layer
            r2_score = model.R2_Score

            results = pd.concat(
                [
                    results,
                    pd.DataFrame({
                        "num_of_layers": [num_of_layer],
                        "mean_num_of_neurons_per_layer": [mean_num_of_neuron_per_layer],
                        "R2": [r2_score],
                    })
                ],
                ignore_index=True
            )

        fig, ax = plotter.create_fig_matplotlib(
            xlabel="Mean Number of Neurons per Layer",
            ylabel="Number of Layers",
            title="R2 Score vs Number of Layers and Mean Number of Neurons per Layer"
        )
        plotter.add_heatmap_to_fig(
            ax=ax,
            data=results,
            title="R2 Score vs Number of Layers and Mean Number of Neurons per Layer"
        )
        plotter.save_plot(fig, path_plots="MLP", filename=f"{file.stem}_plot.png")


if __name__ == "__main__":
    main()