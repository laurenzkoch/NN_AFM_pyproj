"""
LinearRegression
Polynomial Regression
RandomForestRegressor
GradientBoostingRegressor
MLPRegressor aus scikit-learn
"""
from typing import Any

import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error, r2_score

class FIRSTREGRESSION:

    def __init__(self, path_data_motorcad: Path, filename: str) -> None:
        self.path_data_motorcad = path_data_motorcad
        self.filename = filename
        self.data = pd.read_csv(self.path_data_motorcad / self.filename, encoding='utf-8', sep=";", decimal=",")
        self.LRM = LinearRegression()
        self.RFR = RandomForestRegressor()
        self.GBR = GradientBoostingRegressor()
        self.NNR = MLPRegressor()
        self.models = [self.LRM, self.RFR, self.GBR, self.NNR]


    @staticmethod
    def training(x: pd.DataFrame, y: pd.Series, model: Any) -> tuple[float, float]:
        model.fit(x, y)
        y_pred = model.predict(x)
        r2 = r2_score(y, y_pred)
        mean_squared_error_value = mean_squared_error(y, y_pred)
        return mean_squared_error_value, r2

    def train_models(self) -> None:
        x = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        for model in self.models:
            mean_squared_error_value, r2 = self.training(x_train, y_train, model)

            print(f"{model.__class__.__name__}:")
            print(f"Mean squared error: {mean_squared_error_value:.4f}")
            print(f"R^2 score: {r2:.4f}")
            print()

    def plotting(self) -> None:
        self.plot_data()
        self.plot_prediction_peakcurrent()
        self.plot_prediction_phaseadvance()


    def plot_data(self) -> None:
       fig = px.scatter_3d(
           self.data,
           x="PeakCurrent",
           y="PhaseAdvance",
           z="ShaftTorque",
           color="PhaseAdvance"
       )
       fig.show()


    def plot_prediction_peakcurrent(self) -> None:
        fig = plt.figure()
        ax = fig.add_subplot()

        ax.scatter(
            self.data["PeakCurrent"],
            self.data["ShaftTorque"],
            c="blue",
            marker="o",
            label="Actual Shaft Torque"
        )

        ax.scatter(
            self.x_test["PeakCurrent"],
            self.y_pred,
            c="red",
            marker="x",
            label="Predicted Shaft Torque"
        )

        ax.set_xlabel("Peak Current")
        ax.set_ylabel("Shaft Torque")
        ax.set_title("Linear Regression Prediction vs Actual - Peak Current")
        ax.legend()
        plt.show()


    def plot_prediction_phaseadvance(self) -> None:
        fig = plt.figure()
        ax = fig.add_subplot()

        ax.scatter(
            self.data["PhaseAdvance"],
            self.data["ShaftTorque"],
            c="blue",
            marker="o",
            label="Actual Shaft Torque"
        )

        ax.scatter(
            self.x_test["PhaseAdvance"],
            self.y_pred,
            c="red",
            marker="x",
            label="Predicted Shaft Torque"
        )

        ax.set_xlabel("Phase Advance")
        ax.set_ylabel("Shaft Torque")
        ax.set_title("Linear Regression Prediction vs Actual - Phase Advance")
        ax.legend()
        plt.show()

