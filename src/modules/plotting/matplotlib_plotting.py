
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd
from typing import Any

from config.config_paths import *

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


def add_scatter_to_fig(
        ax :Axes,
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


def add_legend_to_fig(ax: Axes, loc: str = "upper left") -> Axes:
    ax.legend(
        loc=loc,
        bbox_to_anchor=(1.1, 1.0),
        borderaxespad=0.0
    )
    return ax

def save_plot(fig: Figure, path_plots: str | Path, filename: str):
    path_plots = PATH_DATA_PLOTS / path_plots
    path_plots.mkdir(exist_ok=True, parents=True)
    fig.savefig(path_plots / filename, bbox_inches="tight", dpi=300)
    plt.close(fig)