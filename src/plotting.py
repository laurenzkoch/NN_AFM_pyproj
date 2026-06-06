from typing import Any

import plotly.express as px
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pandas import DataFrame
import pandas as pd
from pathlib import Path
from matplotlib.figure import Figure


class Plotter:

    def plot_datapoints_plotly(
            self,
            x_column: pd.Series,
            y_column: pd.Series,
            z_column: pd.Series | None = None,
            title: str | None = None,
            xlabel: str | None = None,
            ylabel: str | None = None,
            zlabel: str | None = None,
    ) -> None:

        if z_column is None:
            fig = px.scatter(
                x=x_column,
                y=y_column,
                title=title,
                labels={
                    "x": xlabel,
                    "y": ylabel,
                }
            )
            fig.update_layout(
                xaxis_title=xlabel,
                yaxis_title=ylabel,
            )
        else:
            fig = px.scatter_3d(
                x=x_column,
                y=y_column,
                z=z_column,
                color=z_column,
                title=title,
                labels={
                    "x": xlabel,
                    "y": ylabel,
                    "z": zlabel,
                    "color": zlabel or z_column.name,
                }
            )
            fig.update_layout(
                scene=dict(
                    xaxis_title=xlabel,
                    yaxis_title=ylabel,
                    zaxis_title=zlabel,
                )
            )

        fig.show(renderer="plotly_mimetype")


    def create_fig_matplotlib(
            self,
            xlabel: str,
            ylabel: str,
            title: str,
    ) -> tuple[Figure, Axes]:

        fig, ax = plt.subplots()

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        return fig, ax


    def add_data_to_plot(
            self,
            ax:Axes,
            x_column: pd.Series | Any,
            y_column: pd.Series | Any,
            colors: str,
            marker: str,
            label: str
    ) -> Axes:

        ax.scatter(
            x = x_column,
            y = y_column,
            c=colors,
            marker=marker,
            label=label
        )

        return ax


    def add_text_to_plot(
            self,
            ax: Axes,
            text: str
    ) -> Axes:

        ax.text(
            x=0.02,
            y=0.82,
            s=text,
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
        return ax

    def save_plot(self, fig: Figure, path_plots: str | Path, filename: str):
        path_plots = Path(__file__).resolve().parent.parent / path_plots
        path_plots.mkdir(exist_ok=True)
        fig.savefig(path_plots / filename)