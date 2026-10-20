import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.table import Table
from numpy.typing import NDArray
from torch import Tensor

from src.modules.neuronal_networks.evaluation_io import EvaluationBundle
from src.modules.neuronal_networks.model_io import JSONValue

CHANNEL_LABELS: dict[str, tuple[str, str]] = {
    "ShaftTorque": ("Drehmoment", "Nm"),
    "SystemEfficiency": ("Systemwirkungsgrad", "%"),
    "BMean_AirGap": ("Mittlere Luftspaltflussdichte", "T"),
    "ConductorLoss": ("Wicklungsverluste", "W"),
    "MagnetLoss": ("Magnetverluste", "W"),
    "StatorIronLoss_Total": ("Statoreisenverluste", "W"),
    "RotorIronLoss_Total": ("Rotoreisenverluste", "W"),
}

PREDICTION_STYLE: dict[str, object] = {
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "figure.titlesize": 11,
    "figure.titleweight": "normal",
    "axes.formatter.use_mathtext": False,
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "normal",
    "axes.titlepad": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (7.2, 4.8),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "0.25",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "0.88",
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "figure.constrained_layout.use": True,
    "text.usetex": False,
    "pdf.fonttype": 42,
}


def format_channel_metric(
    metrics: dict[str, JSONValue],
    metric_name: str,
    channel_index: int,
) -> str:
    values: JSONValue = metrics[metric_name]

    if not isinstance(values, list):
        raise TypeError(f"Metric '{metric_name}' must contain a list.")

    value: JSONValue = values[channel_index]

    if value is None:
        return "undefiniert"

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"Metric '{metric_name}', channel {channel_index}: "
            "expected a number or None."
        )

    if metric_name.startswith("r2"):
        return f"{float(value):.3f}"

    return f"{float(value):.3g}"


@dataclass(frozen=True)
class DatasetStyle:
    split_name: Literal["train", "validation", "test"]
    label: str
    color: str
    marker: str
    alpha: float
    marker_size: float


DATASET_STYLES: tuple[DatasetStyle, ...] = (
    DatasetStyle(
        split_name="train",
        label="Training",
        color="#999999",
        marker="o",
        alpha=0.30,
        marker_size=10.0,
    ),
    DatasetStyle(
        split_name="validation",
        label="Validation",
        color="#0065bd",
        marker="^",
        alpha=0.65,
        marker_size=18.0,
    ),
    DatasetStyle(
        split_name="test",
        label="Test",
        color="#e37222",
        marker="s",
        alpha=0.65,
        marker_size=18.0,
    ),
)


def format_mean_r2(value: JSONValue) -> str:
    """Format the mean R-squared consistently for all outputs."""
    if value is None:
        return "undefiniert"

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Metric 'r2_mean' must be a number or None.")

    if not math.isfinite(value):
        return "undefiniert"

    return f"{float(value):.3f}"


def draw_prediction_axis(
    ax: Axes,
    evaluation: EvaluationBundle,
    channel_index: int,
    *,
    include_counts: bool = False,
) -> None:
    """Draw a prediction comparison using the shared presentation rules."""
    channel_name: str = evaluation["channel_names"][channel_index]

    if channel_name not in CHANNEL_LABELS:
        raise ValueError(f"No label defined for channel: {channel_name}")

    unit: str = CHANNEL_LABELS[channel_name][1]

    datasets: list[
        tuple[
            DatasetStyle,
            NDArray[np.float64],
            NDArray[np.float64],
            int,
        ]
    ] = []

    for required_split in ("train", "validation"):
        if required_split not in evaluation["splits"]:
            raise ValueError(f"Required split is missing: {required_split}")

    for style in DATASET_STYLES:
        split = evaluation["splits"].get(style.split_name)

        if split is None:
            continue

        reference: Tensor = split["reference_physical"]
        prediction: Tensor = split["prediction_physical"]

        if (
            reference.shape != prediction.shape
            or reference.ndim != 5
            or reference.shape[-1] != len(evaluation["channel_names"])
            or reference.numel() == 0
        ):
            raise ValueError(f"Invalid evaluation tensor shape: {style.split_name}.")

        machine_count: int = reference.shape[0]

        reference_points: NDArray[np.float64] = (
            reference[..., channel_index].detach().cpu().double().numpy().ravel()
        )
        predicted_points: NDArray[np.float64] = (
            prediction[..., channel_index].detach().cpu().double().numpy().ravel()
        )

        if not (
            bool(np.isfinite(reference_points).all())
            and bool(np.isfinite(predicted_points).all())
        ):
            raise ValueError(f"Non-finite values: {style.split_name}, {channel_name}.")

        datasets.append((style, reference_points, predicted_points, machine_count))

    all_points: NDArray[np.float64] = np.concatenate(
        [
            points
            for _, reference_points, predicted_points, _ in datasets
            for points in (reference_points, predicted_points)
        ]
    )
    minimum: float = float(all_points.min())
    maximum: float = float(all_points.max())
    value_range: float = maximum - minimum

    padding: float = (
        0.05 * value_range if value_range > 0.0 else 0.05 * max(abs(minimum), 1.0)
    )
    limits: tuple[float, float] = (
        minimum - padding,
        maximum + padding,
    )

    ax.set_xlabel(f"MotorCAD-Referenz / {unit}")
    ax.set_ylabel(f"Modellvorhersage / {unit}")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")

    ax.plot(
        limits,
        limits,
        color="0.35",
        linestyle="--",
        linewidth=1.0,
        label="Ideale Vorhersage",
        zorder=1,
    )

    for dataset_index, (
        style,
        reference_points,
        predicted_points,
        machine_count,
    ) in enumerate(datasets):
        legend_label: str = style.label

        if include_counts:
            legend_label = (
                f"{style.label}: {machine_count} Maschinen\n"
                f"{reference_points.size} Kennfeldpunkte"
            )

        ax.scatter(
            reference_points,
            predicted_points,
            label=legend_label,
            color=style.color,
            marker=style.marker,
            s=style.marker_size,
            alpha=style.alpha,
            edgecolors="none",
            zorder=2 + dataset_index,
            rasterized=True,
        )


def draw_channel_metrics(
    ax: Axes,
    evaluation: EvaluationBundle,
    channel_index: int,
    *,
    include_mean: bool,
    bbox: tuple[float, float, float, float],
) -> Table:
    """Draw channel metrics using the shared table formatting."""
    channel_name: str = evaluation["channel_names"][channel_index]
    unit: str = CHANNEL_LABELS[channel_name][1]
    rows: list[list[str]] = []

    for style in DATASET_STYLES:
        split = evaluation["splits"].get(style.split_name)

        if split is None:
            rows.append(
                [style.label, "—", "—", "—"]
                if include_mean
                else [style.label, "—", "—"]
            )
            continue

        metrics: dict[str, JSONValue] = split["metrics"]
        row: list[str] = [
            style.label,
            format_channel_metric(
                metrics,
                "r2_per_channel",
                channel_index,
            ),
        ]

        if include_mean:
            row.append(format_mean_r2(metrics["r2_mean"]))

        row.append(
            format_channel_metric(
                metrics,
                "mae_physical_per_channel",
                channel_index,
            )
        )
        rows.append(row)

    headers: list[str] = ["Datensatz", "R²"]

    if include_mean:
        headers.append("R² mean")

    headers.append(f"MAE / {unit}")

    widths: list[float] = (
        [0.30, 0.19, 0.23, 0.28] if include_mean else [0.44, 0.23, 0.33]
    )

    table: Table = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=widths,
        cellLoc="right",
        colLoc="right",
        bbox=bbox,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("0.75")
        cell.set_linewidth(0.6)
        cell.visible_edges = "B" if row_index == 0 else ""

        if row_index == 0:
            cell.get_text().set_fontweight("bold")

        if column_index == 0:
            cell.get_text().set_horizontalalignment("left")

    return table
