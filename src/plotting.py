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
    ) -> None:

        if z_column is None:
            fig = px.scatter(
                x=x_column,
                y=y_column
            )
        else:
            fig = px.scatter_3d(
                x=x_column,
                y=y_column,
                z=z_column,
                color=z_column
            )

        fig.show()


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
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            borderaxespad=0.0
        )
        plt.show()

        return fig, ax


    def add_data_to_plot(
            self,
            ax:Axes,
            x_column: pd.Series,
            y_column: pd.Series,
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


    def add_text_to_plot(self, ax: Axes, text: str) -> None:

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


    def save_plot(self, fig: Figure, filename: str):
        path_plots = Path(__file__).resolve().parent.parent / "plots"
        path_plots.mkdir(exist_ok=True)
        fig.savefig(path_plots / filename)