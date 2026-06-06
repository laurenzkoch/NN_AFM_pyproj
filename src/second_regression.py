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

from src import ml_model_trainig, data_managing, plotting


class SECONDREGRESSION(ml_model_trainig.MLModelTrainer, data_managing.DataManager):



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
        for split in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            train_models_df = self.train_models(test_split=split, x=x, y=y, models=models)


    def train_models(self, test_split, x, y, models: list[Any]) -> pd.DataFrame:

        splits = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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
            trained_and_evaluated_model = self.evaluate_model(model=model, x_test=x_test, y_test=y_test)
            for column in x.columns:
                fig, ax = self.Plotter.create_fig_matplotlib(column, y.columns[0], title=f"Model: {model.__class__.__name__}")
                ax = self.Plotter.add_data_to_plot(ax, x[column], y.iloc[:,0], colors="b", marker="o", label="Simulated Data")
                ax = self.Plotter.add_data_to_plot(ax, x_test[column], trained_and_evaluated_model["y_pred"].iloc[0], colors="r", marker="x", label=model.__class__.__name__)
                ax = self.Plotter.add_text_to_plot(
                    ax,
                    text=f"R^2 Score: {trained_and_evaluated_model["R^2 Score"].iloc[0]:.2f}\nMean Squared Error: {trained_and_evaluated_model["Mean Squared Error"].iloc[0]:.2f}\nSplit: {test_split:.2f}",
                )

                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(0.02, 0.98),
                    borderaxespad=0.0
                )
                plt.show()

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