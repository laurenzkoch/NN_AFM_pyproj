"""
LinearRegression
Polynomial Regression
RandomForestRegressor
GradientBoostingRegressor
MLPRegressor aus scikit-learn
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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error, r2_score

class FIRSTREGRESSION:

    def __init__(self, path_data_motorcad: Path, filename: str) -> None:
        self.data, self.x_train, self.x_test, self.y_train, self.y_test = self.import_data(path_data_motorcad, filename)
        self.plot_data()

        lrm = LinearRegression()
        rfr = RandomForestRegressor()
        gbr = GradientBoostingRegressor()
        nnr = MLPRegressor()
        self.models = [lrm, rfr, gbr, nnr]
        self.prediction = self.train_models()
        self.plot_prediction()


    @staticmethod
    def training(x: pd.DataFrame, y: pd.Series, model: Any) -> tuple[np.ndarray, float, float]:
        model.fit(x, y)
        y_pred = model.predict(x)
        r2 = r2_score(y, y_pred)
        mean_squared_error_value = cast(float, mean_squared_error(y, y_pred))
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
            y_pred, mean_squared_error_value, r2 = self.training(self.x_train, self.y_train, model)

            prediction_df = pd.concat(
                [
                    prediction_df,
                    pd.DataFrame([{
                        "y_pred": y_pred,
                        "Model": model.__class__.__name__,
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

        colors = ["red", "green", "orange", "purple"]
        for color_index, (index, model_values) in enumerate(self.prediction.iterrows()):
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
                    self.x_train.iloc[:, colum],
                    model_values["y_pred"],
                    c=colors[color_index],
                    marker="x",
                    label=model_values["Model"]
                )

                ax.set_xlabel(self.data.columns[colum])
                ax.set_ylabel("Shaft Torque")
                ax.set_title(f"Linear Regression Prediction vs Actual - {self.data.columns[colum]}")
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

                filename = f"plot_shafttorque_predicted_over_{self.data.columns[colum]}_{model_values['Model']}.png"
                utils.save_plot(fig, filename)

                plt.show()









