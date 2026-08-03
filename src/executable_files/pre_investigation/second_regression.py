"""- RandomForestRegressor
- GradientBoostingRegressor
- HistGradientBoostingRegressor
- GaussianProcessRegressor
- MLPRegressor"""

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from typing import Any

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor

from modules.utilitis import plotting, data_managing, ml_model_training


class SECONDREGRESSION(ml_model_training.MLModelTrainer, data_managing.DataManager):



    def __init__(self, path_data: Path, filename: str) -> None:
        super().__init__()
        self.Plotter = plotting.Plotter()
        self.path_plots = Path(__file__).resolve().parent.parent / "plots" / "second_reg"

        rfr = RandomForestRegressor()
        gbr = GradientBoostingRegressor()
        hgbr = HistGradientBoostingRegressor()
        gpr = GaussianProcessRegressor()
        nnr = MLPRegressor(
            max_iter=2000,
            random_state=42,
            early_stopping=True
        )

        models = [rfr, gbr, hgbr, gpr, nnr]

        data_df = self.import_data_from_csv(path_data, filename)
        x, y = self.split_data(data_df=data_df, number_of_outputs=1)
        self.Plotter.plot_datapoints_plotly(
            x.iloc[:, 0],
            x.iloc[:, 1],
            y.iloc[:, 0],
            title="Data Points",
            xlabel="Peak Current [A]",
            ylabel="Phase Advance [Degrees]",
            zlabel="Shaft Torque [Nm]"
        )

        train_models_splits = []
        splits = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for split in splits:
            train_models_df = self.train_models(test_split=split, x=x, y=y, models=models)
            train_models_splits.append(train_models_df)


        fig, ax = self.Plotter.create_fig_matplotlib(xlabel="Test Split", ylabel="R^2 Score", title="R^2 Score")
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for model in range(len(models)):
            model_name = train_models_splits[0].iloc[model]["Model"]
            ax = self.Plotter.add_scatter_to_fig(ax, splits, [train_models_splits[i].iloc[model]["R^2 Score"] for i in
                                                              range(len(splits))], colors=color_cycle[model],
                                                 marker="o", label=model_name)
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.02, 0.02),
            borderaxespad=0.0
        )
        filename = f"plot_R2_over_Split.png"
        self.Plotter.save_plot(fig, path_plots=self.path_plots, filename=filename)

        fig, ax = self.Plotter.create_fig_matplotlib(xlabel="Test Split", ylabel="MSE", title="MSE")
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for model in range(len(models)):
            model_name = train_models_splits[0].iloc[model]["Model"]
            ax = self.Plotter.add_scatter_to_fig(ax, splits,
                                                 [train_models_splits[i].iloc[model]["Mean Squared Error"] for i in
                                                  range(len(splits))], colors=color_cycle[model], marker="o",
                                                 label=model_name)

        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            borderaxespad=0.0
        )
        filename = f"plot_MSE_over_Split.png"
        self.Plotter.save_plot(fig, path_plots=self.path_plots, filename=filename)


    def train_models(self, test_split, x, y, models: list[Any]) -> pd.DataFrame:

        prediction_df = pd.DataFrame(
            columns=[
                "Model",
                "R^2 Score",
                "Mean Squared Error",
                "y_pred",
                "x_test",
            ]
        )

        for model in models:
            model, x_test, y_test = self.train_model(x, y, test_split=test_split, model=model)
            trained_and_evaluated_model = self.evaluate_model(trained_model=model, x_test=x_test, y_test=y_test)
            for column in x.columns:
                fig, ax = self.Plotter.create_fig_matplotlib(column, y.columns[0], title=f"Model: {model.__class__.__name__}")
                ax = self.Plotter.add_scatter_to_fig(ax, x[column], y.iloc[:, 0], colors="b", marker="o",
                                                     label="Simulated Data")
                ax = self.Plotter.add_scatter_to_fig(ax, x_test[column], trained_and_evaluated_model["y_pred"].iloc[0],
                                                     colors="r", marker="x", label=model.__class__.__name__)
                ax = self.Plotter.add_text_to_fig(ax,
                                                  text=f"R^2 Score: {trained_and_evaluated_model["R^2 Score"].iloc[0]:.2f}\nMean Squared Error: {trained_and_evaluated_model["Mean Squared Error"].iloc[0]:.2f}\nSplit: {test_split:.2f}")

                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(0.02, 0.98),
                    borderaxespad=0.0
                )
                plt.close()

                filename = f"plot_shafttorque_predicted_over_{column}_{model.__class__.__name__}_{test_split:.2f}.png"
                self.Plotter.save_plot(fig, path_plots=self.path_plots, filename=filename)

            prediction_df = pd.concat(
                [
                    prediction_df,
                    trained_and_evaluated_model
                ],
                ignore_index=True
            )
        return prediction_df