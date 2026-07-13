from typing import Any

import plotly.express as px
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from src.config import config_paths
import pandas as pd
from pathlib import Path
from matplotlib.figure import Figure
import seaborn as sns


class Plotter:

    @staticmethod
    def plot_datapoints_plotly(
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


    @staticmethod
    def create_fig_matplotlib(
            xlabel: str,
            ylabel: str,
            title: str,
    ) -> tuple[Figure, Axes]:

        fig, ax = plt.subplots()

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        return fig, ax


    @staticmethod
    def add_scatter_to_fig(
            ax:Axes,
            x_column: pd.Series | Any,
            y_column: pd.Series | Any,
            z_column: pd.Series | None = None,
            colors: str = "red",
            marker: str = "x",
            label: str = ""
    ) -> Axes:

        if z_column is not None:
            ax.scatter(
                x=x_column,
                y=y_column,
                c=z_column,
                cmap="viridis",
                marker=marker,
                label=label
            )
        else:
            ax.scatter(
                x=x_column,
                y=y_column,
                c=colors,
                marker=marker,
                label=label
            )

        return ax

    @staticmethod
    def add_plot_to_fig(
            ax: Axes,
            x_column: pd.Series | Any,
            y_column: pd.Series | Any,
            colors: str,
            label: str
    ) -> Axes:

        ax.plot(
            x_column,
            y_column,
            color=colors,
            label=label
        )

        return ax


    @staticmethod
    def add_text_to_fig(
            ax: Axes,
            text: str
    ) -> Axes:

        ax.text(
            x=1.02,
            y=0.8,
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


    @staticmethod
    def add_legend_to_fig(ax: Axes) -> Axes:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0
        )
        return ax


    @staticmethod
    def add_heatmap_to_fig(ax: Axes, data: pd.DataFrame, title: str) -> Axes:

        heatmap_data = data.pivot_table(
            index=data.columns[0],
            columns=data.columns[1],
            values=data.columns[2],
            aggfunc = "mean"
        ).fillna(0).astype(float)

        sns.heatmap(
            heatmap_data,
            annot=False,  # Zahlen in den Zellen anzeigen
            fmt=".2f",  # Zahlenformat
            cmap="viridis",  # Farbpalette
            linewidths=0.0,  # Linien zwischen den Zellen
            linecolor="black",
            cbar_kws={"label": "R^2"},
            ax=ax
        )

        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel(data.columns[1])
        ax.set_ylabel(data.columns[0])

        plt.tight_layout()
        return ax


    @staticmethod
    def save_plot(fig: Figure, path_plots: str | Path, filename: str):
        path_plots = config_paths.PATH_DATA_PLOTS / path_plots
        path_plots.mkdir(exist_ok=True)
        fig.savefig(path_plots / filename, bbox_inches="tight", dpi=300)
        plt.close(fig)