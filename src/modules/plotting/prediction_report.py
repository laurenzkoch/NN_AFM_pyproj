import logging
import math
from pathlib import Path
from textwrap import fill
from typing import cast

import matplotlib as mpl
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.gridspec import SubplotSpec

from src.config.config_paths import PATH_DATA_TRAINED_NN
from src.modules.data_managment.experiment_utils import file_sha256
from src.modules.neuronal_networks.evaluation_io import EvaluationBundle
from src.modules.neuronal_networks.model_io import JSONValue, SurrogateCheckpoint
from src.modules.plotting.prediction_base import (
    CHANNEL_LABELS,
    DATASET_STYLES,
    PREDICTION_STYLE,
    draw_channel_metrics,
    draw_prediction_axis,
    format_mean_r2,
)

logger = logging.getLogger(__name__)


def _load_report_data(
    model_folder: str,
) -> tuple[Path, EvaluationBundle]:
    """Load saved evaluation data and determine the report destination."""
    if (
        not model_folder
        or model_folder in {".", ".."}
        or Path(model_folder).name != model_folder
    ):
        raise ValueError("Please state only the model folder name.")

    model_directory: Path = Path(PATH_DATA_TRAINED_NN) / model_folder
    evaluation_path: Path = model_directory / "evaluation.pt"

    if not evaluation_path.is_file():
        raise FileNotFoundError(f"Evaluation file not found: {evaluation_path}")

    evaluation: EvaluationBundle = cast(
        EvaluationBundle,
        torch.load(
            evaluation_path,
            map_location="cpu",
            weights_only=True,
        ),
    )

    if len(evaluation["channel_names"]) != 7:
        raise ValueError("The report requires exactly seven output channels.")

    for split_name in ("train", "validation"):
        if split_name not in evaluation["splits"]:
            raise ValueError(f"Required evaluation split is missing: {split_name}")

    output_path: Path = model_directory / "result_plots" / "prediction_report.pdf"

    return output_path, evaluation


def _load_report_checkpoint(
    model_folder: str,
    evaluation: EvaluationBundle,
) -> SurrogateCheckpoint:
    """Load the checkpoint belonging to the saved evaluation."""
    model_filename: str = evaluation["model_filename"]

    if not model_filename or Path(model_filename).name != model_filename:
        raise ValueError("Invalid model filename in evaluation data.")

    model_path: Path = Path(PATH_DATA_TRAINED_NN) / model_folder / model_filename

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if file_sha256(model_path) != evaluation["model_sha256"]:
        raise ValueError("The model file does not match the saved evaluation.")

    raw: object = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(raw, dict):
        raise TypeError("The model checkpoint must contain a dictionary.")

    required_fields: set[str] = {
        "model_type",
        "activation",
        "feature_names",
        "channel_names",
        "output_dims",
        "hidden_layers",
        "seed",
        "metrics",
    }
    missing_fields: set[str] = required_fields.difference(raw)

    if missing_fields:
        raise ValueError(f"Missing checkpoint fields: {sorted(missing_fields)}")

    return cast(SurrogateCheckpoint, raw)


def _create_report_page(
    evaluation: EvaluationBundle,
    model_name: str,
) -> tuple[Figure, tuple[SubplotSpec, ...], Axes]:
    """Create one A4 landscape page with summary and seven plot areas."""
    fig: Figure = plt.figure(
        figsize=(297 / 25.4, 210 / 25.4),
        layout="constrained",
    )

    grid = fig.add_gridspec(
        nrows=3,
        ncols=4,
        height_ratios=(0.65, 2.5, 2.5),
        hspace=0.12,
        wspace=0.12,
    )

    fig.suptitle(
        "Bewertung des Ersatzmodells",
        fontsize=11,
        fontweight="bold",
    )

    summary_ax: Axes = fig.add_subplot(grid[0, :])
    summary_ax.set_axis_off()
    summary_ax.set_title(
        f"Modell: {model_name}",
        fontsize=10,
        fontweight="normal",
    )

    summary_rows: list[list[str]] = []

    for style in DATASET_STYLES:
        split = evaluation["splits"].get(style.split_name)

        if split is None:
            summary_rows.append([style.label, "nicht vorhanden", "—"])
            continue

        machine_count: int = len(split["machine_ids"])
        mean_r2: str = format_mean_r2(split["metrics"]["r2_mean"])

        summary_rows.append([style.label, str(machine_count), mean_r2])

    summary_table = summary_ax.table(
        cellText=summary_rows,
        colLabels=["Datensatz", "Maschinen", "R² mean · sieben Kanäle"],
        colWidths=[0.28, 0.22, 0.50],
        cellLoc="center",
        colLoc="center",
        bbox=(0.20, 0.0, 0.60, 0.90),
    )
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(9)

    for (row_index, _), cell in summary_table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("0.75")
        cell.set_linewidth(0.6)
        cell.visible_edges = "B" if row_index == 0 else ""

        if row_index == 0:
            cell.get_text().set_fontweight("bold")

    panel_specs: tuple[SubplotSpec, ...] = tuple(
        grid[1 + index // 4, index % 4] for index in range(7)
    )

    legend_ax: Axes = fig.add_subplot(grid[2, 3])
    legend_ax.set_axis_off()

    return fig, panel_specs, legend_ax


def _draw_report_panel(
    fig: Figure,
    panel_spec: SubplotSpec,
    evaluation: EvaluationBundle,
    channel_index: int,
) -> Axes:
    """Arrange a shared prediction plot and its channel metrics."""
    channel_name: str = evaluation["channel_names"][channel_index]

    if channel_name not in CHANNEL_LABELS:
        raise ValueError(f"No label defined for channel: {channel_name}")

    quantity_name: str = CHANNEL_LABELS[channel_name][0]

    panel_grid = panel_spec.subgridspec(
        2,
        1,
        height_ratios=(3.0, 1.0),
        hspace=0.0,
    )

    ax: Axes = fig.add_subplot(panel_grid[0])
    metrics_ax: Axes = fig.add_subplot(panel_grid[1])
    metrics_ax.set_axis_off()

    draw_prediction_axis(
        ax,
        evaluation,
        channel_index,
        include_counts=False,
    )
    ax.set_anchor("S")
    ax.set_title(
        fill(quantity_name, width=27),
        fontsize=11,
        fontweight="bold",
    )

    draw_channel_metrics(
        metrics_ax,
        evaluation,
        channel_index,
        include_mean=False,
        bbox=(0.0, 0.0, 1.0, 1.0),
    )

    return ax


def _format_training_setting(value: JSONValue) -> str:
    """Format a saved numeric training setting."""
    if value is None:
        return "nicht gespeichert"

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("A training setting must be numeric or None.")

    if isinstance(value, int):
        return str(value)

    if not math.isfinite(value):
        return "undefiniert"

    return f"{value:.4g}"


def _build_model_information(
    checkpoint: SurrogateCheckpoint,
) -> str:
    """Describe the saved model architecture and training settings."""
    training: JSONValue = checkpoint["metrics"].get("training_config")

    if training is None:
        training_settings: dict[str, JSONValue] = {}
    elif isinstance(training, dict):
        training_settings = training
    else:
        raise TypeError("'training_config' must contain a dictionary.")

    layer_sizes: tuple[int, ...] = (
        len(checkpoint["feature_names"]),
        *checkpoint["hidden_layers"],
        math.prod(checkpoint["output_dims"]),
    )
    architecture: str = " → ".join(str(size) for size in layer_sizes)
    model_count: int = len(checkpoint["channel_names"])

    learning_rate: str = _format_training_setting(
        training_settings.get("learning_rate")
    )
    weight_decay: str = _format_training_setting(training_settings.get("weight_decay"))
    maximum_epochs: str = _format_training_setting(training_settings.get("num_epochs"))
    patience: str = _format_training_setting(training_settings.get("patience"))

    return "\n".join(
        [
            f"Modell: {model_count} MLP-Submodelle",
            f"Architektur je MLP:\n{architecture}",
            f"Aktivierung: {checkpoint['activation']}",
            f"Lernrate: {learning_rate}",
            f"Weight Decay: {weight_decay}",
            f"Max. Epochen: {maximum_epochs}",
            f"Early-Stopping-Patience: {patience}",
        ]
    )


@mpl.rc_context(rc={**PREDICTION_STYLE, "savefig.bbox": None})
def create_prediction_report(model_folder: str) -> Path:
    """Save a single-page PDF report containing all seven output channels."""
    findfont(
        FontProperties(family="Arial"),
        fallback_to_default=False,
    )

    output_path: Path
    evaluation: EvaluationBundle
    output_path, evaluation = _load_report_data(model_folder)
    checkpoint: SurrogateCheckpoint = _load_report_checkpoint(
        model_folder,
        evaluation,
    )
    model_information: str = _build_model_information(checkpoint)

    fig: Figure
    panel_specs: tuple[SubplotSpec, ...]
    legend_ax: Axes
    fig, panel_specs, legend_ax = _create_report_page(
        evaluation,
        model_folder,
    )

    try:
        plot_axes: list[Axes] = [
            _draw_report_panel(fig, spec, evaluation, channel_index)
            for channel_index, spec in enumerate(panel_specs)
        ]

        handles, labels = plot_axes[0].get_legend_handles_labels()
        legend_ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            borderaxespad=0.0,
            frameon=False,
            fontsize=9,
            labelspacing=0.9,
            handlelength=1.6,
            markerscale=1.4,
        )

        legend_ax.text(
            0.0,
            0.62,
            "Modell und Training",
            transform=legend_ax.transAxes,
            fontsize=10,
            fontweight="bold",
            verticalalignment="top",
        )

        legend_ax.text(
            0.0,
            0.53,
            model_information,
            transform=legend_ax.transAxes,
            fontsize=9,
            horizontalalignment="left",
            verticalalignment="top",
            linespacing=1.1,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            output_path,
            format="pdf",
            dpi=300,
            facecolor="white",
            metadata={
                "Title": "Bewertung des Ersatzmodells",
                "Subject": model_folder,
            },
        )
    finally:
        plt.close(fig)

    logger.info("Prediction report saved: %s", output_path)
    return output_path
