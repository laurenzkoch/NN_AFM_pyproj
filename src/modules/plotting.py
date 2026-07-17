from typing import Any
from collections.abc import Sequence
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
            colors: str | tuple[float, float, float, float] = "r",
            label: str = ""
    ) -> Axes:

        ax.plot(
            x_column,
            y_column,
            color=colors,
            label=label
        )
        ax.grid(
            True,
            linestyle="--",
            linewidth=0.5,
            alpha=0.5
        )

        return ax


    @staticmethod
    def add_text_to_fig(
            ax: Axes,
            text: str,
            x: float = 1.1,
            y: float = 0.8,
    ) -> Axes:

        ax.text(
            x=x,
            y=y,
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
    def add_legend_to_fig(ax: Axes, loc: str = "upper left") -> Axes:
        ax.legend(
            loc=loc,
            bbox_to_anchor=(1.1, 1.0),
            borderaxespad=0.0
        )
        return ax


    @staticmethod
    def add_heatmap_to_fig(
            ax: Axes,
            label_x: str,
            label_y: str,
            label_z: str,
            x: pd.Series | Sequence[Any],
            y: pd.Series | Sequence[Any],
            z: pd.Series | Sequence[float],
            title: str
    ) -> Axes:

        data = pd.DataFrame({
            "x": x,
            "y": y,
            "z": z
        })

        heatmap_data = data.pivot_table(
            index="y",
            columns="x",
            values="z",
            aggfunc="mean"
        ).astype(float)

        heatmap_data = (
            heatmap_data
            .interpolate(
                method="linear",
                axis=0,
                limit_direction="both"
            )
            .interpolate(
                method="linear",
                axis=1,
                limit_direction="both"
            )
        )

        sns.heatmap(
            heatmap_data,
            annot=False,
            fmt=".2f",
            cmap="viridis",
            linewidths=0.0,
            linecolor="black",
            cbar_kws={"label": label_z},
            ax=ax
        )
        ax.set_xticklabels(
            [
                f"{float(label.get_text()):.2f}".rstrip("0").rstrip(".")
                for label in ax.get_xticklabels()
            ],
            rotation=90
        )

        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)

        ax.figure.tight_layout()

        return ax


    @staticmethod
    def save_plot(fig: Figure, path_plots: str | Path, filename: str):
        path_plots = config_paths.PATH_DATA_PLOTS / path_plots
        path_plots.mkdir(exist_ok=True, parents=True)
        fig.savefig(path_plots / filename, bbox_inches="tight", dpi=300)
        plt.close(fig)