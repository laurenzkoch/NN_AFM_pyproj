from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm


def add_heatmap_to_fig(
        ax: Axes,
        label_x: str,
        label_y: str,
        label_z: str,
        x: pd.Series | Sequence[Any],
        y: pd.Series | Sequence[Any],
        z: pd.Series | Sequence[float],
        title: str,
        log_scale: bool = False,
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
    vmin = 0
    vmax = 1
    if log_scale:

        positive_values = heatmap_data.to_numpy()
        positive_values = positive_values[
            np.isfinite(positive_values) & (positive_values > 0)
            ]

        if positive_values.size == 0:
            raise ValueError(
                "Für eine logarithmische Farbskala muss mindestens "
                "ein positiver z-Wert vorhanden sein."
            )

        vmin = positive_values.min()
        vmax = positive_values.max()
        heatmap_data = heatmap_data.mask(heatmap_data <= 0)

    sns.heatmap(
        heatmap_data,
        annot=False,
        fmt=".2f",
        cmap="viridis",
        norm=LogNorm(
            vmin=vmin,
            vmax=vmax
        ) if log_scale else None,
        linewidths=0.0,
        linecolor="black",
        cbar_kws={
            "label": label_z
        },
        ax=ax
    )

    ax.set_xticklabels(
        [
            f"{float(label.get_text()):.2f}".rstrip("0").rstrip(".")
            for label in ax.get_xticklabels()
        ],
        rotation=90
    )

    ax.set_yticklabels(
        [
            f"{float(label.get_text()):.2f}".rstrip("0").rstrip(".")
            for label in ax.get_yticklabels()
        ],
        rotation=0
    )

    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)

    ax.figure.tight_layout()

    return ax
