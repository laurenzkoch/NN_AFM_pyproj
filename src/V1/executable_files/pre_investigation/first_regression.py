"""
LinearRegression: linear > shit
RandomForestRegressor: sehr gut
GradientBoostingRegressor: befriedigend bis gut
HistGradientBoostingRegressor: befriedigend bis gut
GaussianProcessRegressor: ausreichend
MLPRegressor: Mangelhaft > Potential ?
Polynomial Regression: nur als Basis empfehlenswert
SVR: mangelhaft bis ausreichend > evtl. Feature Engineering
Ridge: linear > shit
Lasso: linear > shit
ElasticNet: weitgehend linear > shit



für weitere Untersuchungen:
- RandomForestRegressor
- GradientBoostingRegressor
- HistGradientBoostingRegressor
- GaussianProcessRegressor
- MLPRegressor
"""

import utils

import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
from matplotlib import pyplot as plt
from pandas import DataFrame
from typing import Any, cast

from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, r2_score

class FIRSTREGRESSION:

    def __init__(self, path_data_motorcad: Path, filename: str) -> None:
        self.data, self.x_train, self.x_test, self.y_train, self.y_test = self.import_data(path_data_motorcad, filename)
        self.plot_data()

        linearregression = LinearRegression()
        randomforest = RandomForestRegressor()
        gradientboosting = GradientBoostingRegressor()
        histgradientboosting = HistGradientBoostingRegressor()
        gaussianprocess = GaussianProcessRegressor()
        nnr = MLPRegressor()
        poly_linear = make_pipeline(
            PolynomialFeatures(degree=3, include_bias=False),
            LinearRegression()
        )

        poly_ridge = make_pipeline(
            PolynomialFeatures(degree=3, include_bias=False),
            Ridge(alpha=1.0)
        )

        poly_lasso = make_pipeline(
            PolynomialFeatures(degree=3, include_bias=False),
            Lasso(alpha=0.001, max_iter=10000)
        )

        poly_elastic = make_pipeline(
            PolynomialFeatures(degree=3, include_bias=False),
            ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
        )
        svr = SVR()
        ridge = Ridge()
        lasso = Lasso()
        elastic = ElasticNet()

        self.models = [
            linearregression,
            randomforest,
            gradientboosting,
            nnr,
            histgradientboosting,
            gaussianprocess,
            poly_linear,
            poly_ridge,
            poly_lasso,
            poly_elastic,
            svr,
            ridge,
            lasso,
            elastic
        ]
        self.prediction = self.train_models()
        self.plot_prediction()


    @staticmethod
    def training(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, model: Any) -> tuple[np.ndarray, float, float]:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        mean_squared_error_value = cast(float, mean_squared_error(y_test, y_pred))
        return y_pred, mean_squared_error_value, r2

    @staticmethod
    def import_data(path_data_motorcad: Path, filename: str) -> tuple[DataFrame, Any, Any, Any, Any]:
        data = pd.read_csv(path_data_motorcad / filename, encoding='utf-8', sep=";", decimal=",")
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        return data, x_train, x_test, y_train, y_test


    def train_models(self) -> pd.DataFrame:
        prediction_df = pd.DataFrame(columns=['y_pred','Model', 'Mean Squared Error', 'R^2 Score'])

        for model in self.models:
            y_pred, mean_squared_error_value, r2 = self.training(self.x_train, self.x_test, self.y_train, self.y_test, model)

            prediction_df = pd.concat(
                [
                    prediction_df,
                    pd.DataFrame([{
                        "y_pred": y_pred,
                        "Model": model,
                        "Mean Squared Error": mean_squared_error_value,
                        "R^2 Score": r2,
                    }])
                ],
                ignore_index=True
            )
        return prediction_df

    def plot_data(self) -> None:
        fig = px.scatter_3d(
            self.data,
            x="PeakCurrent",
            y="PhaseAdvance",
            z="ShaftTorque",
            color="PhaseAdvance"
        )
        fig.show()


    def plot_prediction(self) -> None:

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        units = ["A", "rad"]
        for color_index, (index, model_values) in enumerate(self.prediction.iterrows()):
            model_name = model_values["Model"].__class__.__name__
            if model_name == "Pipeline":
                model_name = "Pipeline " + model_values["Model"].steps[-1][1].__class__.__name__
            model_color = color_cycle[color_index % len(color_cycle)]

            for colum in range(len(self.data.columns) - 1):
                fig = plt.figure()
                ax = fig.add_subplot()
                ax.scatter(
                    self.data.iloc[:, colum],
                    self.data.iloc[:, -1],
                    c="blue",
                    marker="o",
                    label="Actual Shaft Torque"
                )
                ax.scatter(
                    self.x_test.iloc[:, colum],
                    model_values["y_pred"],
                    c=model_color,
                    marker="x",
                    label=model_name
                )

                ax.set_xlabel(f"{self.data.columns[colum]} [{units[colum]}]")
                ax.set_ylabel("Shaft Torque [Nm]")
                ax.set_title(f"{model_name} Prediction vs Actual Shaft Torque")
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(0.02, 0.98),
                    borderaxespad=0.0
                )
                ax.text(
                    0.02,
                    0.82,
                    f"R2: {model_values['R^2 Score']:.2f}\nMSE: {model_values['Mean Squared Error']:.2f}",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="black",
                    verticalalignment="top",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="black",
                        alpha=0.8
                    )
                )

                filename = f"plot_shafttorque_predicted_over_{self.data.columns[colum]}_{model_name}.png"
                utils.save_plot(fig, filename)

                plt.show()









