"""- RandomForestRegressor
- GradientBoostingRegressor
- HistGradientBoostingRegressor
- GaussianProcessRegressor
- MLPRegressor"""

import utils

import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
from matplotlib import pyplot as plt
from pandas import DataFrame
from typing import Any, cast

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor

from src import ml_model_trainig, data_managing, plotting


class SECONDREGRESSION(ml_model_trainig.MLModelTrainer, data_managing.DataManager, plotting.Plotter):

    def __init__(self, path_data: Path, filename: str) -> None:
        super().__init__()
        rfr = RandomForestRegressor()
        gbr = GradientBoostingRegressor()
        hgbr = HistGradientBoostingRegressor()
        gpr = GaussianProcessRegressor()
        nnr = MLPRegressor()

        models = [rfr, gbr, hgbr, gpr, nnr]

        data_df = self.import_data_from_csv(path_data, filename)
        x, y = self.split_data(data_df=data_df, number_of_outputs=1)
        train_models_df = self.train_models(test_split=0.2, x=x, y=y, models=models)
        self.plot_datapoints_plotly(x.iloc[:, 0], x.iloc[:, 1], y.iloc[:, 0])


    def train_models(self, test_split: float, x, y, models: list[Any]) -> pd.DataFrame:
        prediction_df = pd.DataFrame(
            columns=[
                "Model",
                "R^2 Score",
                "Mean Squared Error",
                "y_pred",
                "x_test"
            ]
        )

        for model in models:
            model, x_test, y_test = self.train_model(x, y, test_split=test_split, model=model)
            trained_and_evaluated_model = self.evaluate_model(model=model, x_test=x_test, y_test=y_test)

            self.create_plotly_scatter(x_test, y_test, trained_and_evaluated_model["y_pred"], model.__class__.__name__)

            prediction_df = pd.concat(
                [
                    prediction_df,
                    trained_and_evaluated_model
                ],
                ignore_index=True
            )
        return prediction_df