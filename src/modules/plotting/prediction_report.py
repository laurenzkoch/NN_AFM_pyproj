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
from matplotlib.table import Table
from matplotlib.table import table as create_table
from matplotlib.transforms import Bbox

from src.modules.data_managment.experiment_utils import file_sha256
from src.modules.neuronal_networks.model_data_io.autoencoder_io import (
    AutoencoderCollectionCheckpoint,
    load_autoencoders,
)
from src.modules.neuronal_networks.model_data_io.evaluation_io import EvaluationBundle
from src.modules.neuronal_networks.model_data_io.model_io import (
    JSONValue,
    SurrogateCheckpointV2,
    load_bundle,
)
from src.modules.neuronal_networks.model_data_io.model_paths import (
    resolve_model_directory,
)
from src.modules.neuronal_networks.model_data_io.network_io import (
    AutoencoderCheckpoint,
    checked_training_duration,
)
from src.modules.plotting.prediction_base import (
    CHANNEL_LABELS,
    DATASET_STYLES,
    PREDICTION_STYLE,
    draw_channel_metrics,
    draw_prediction_axis,
    format_mean_r2,
)

type ReportCheckpoint = SurrogateCheckpointV2 | AutoencoderCollectionCheckpoint

logger = logging.getLogger(__name__)


def _load_report_data(
    model_folder: str | Path,
) -> tuple[Path, EvaluationBundle]:

    model_directory: Path = resolve_model_directory(model_folder)
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
    model_folder: str | Path,
    evaluation: EvaluationBundle,
) -> ReportCheckpoint:
    """Load the complete model belonging to the saved evaluation."""
    model_directory: Path = resolve_model_directory(model_folder)
    model_filename: str = evaluation["model_filename"]

    if (
        not model_filename
        or model_filename in {".", ".."}
        or Path(model_filename).name != model_filename
    ):
        raise ValueError("Invalid model filename in evaluation data.")

    model_path: Path = model_directory / model_filename

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if file_sha256(model_path) != evaluation["model_sha256"]:
        raise ValueError("The model file does not match the saved evaluation.")

    if evaluation["evaluation_kind"] == "reconstruction":
        _, cnn_checkpoint = load_autoencoders(model_path)
        return cnn_checkpoint

    if evaluation["evaluation_kind"] == "prediction":
        _, surrogate_checkpoint = load_bundle(model_path)
        return surrogate_checkpoint

    raise ValueError("Unsupported evaluation kind.")


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

    report_title: str = (
        "Bewertung der CNN-Rekonstruktion"
        if evaluation["evaluation_kind"] == "reconstruction"
        else "Bewertung des Ersatzmodells"
    )

    fig.suptitle(
        report_title,
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

    summary_table: Table = create_table(
        ax=summary_ax,
        cellText=summary_rows,
        colLabels=["Datensatz", "Maschinen", "R² mean · sieben Kanäle"],
        colWidths=[0.28, 0.22, 0.50],
        cellLoc="center",
        colLoc="center",
        bbox=Bbox.from_bounds(0.20, 0.0, 0.60, 0.90),
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


def _channel_architecture_text(
    checkpoint: ReportCheckpoint,
    channel_index: int,
) -> str:
    """Describe the networks belonging to one output channel."""
    lines: list[str] = []
    cnn: AutoencoderCheckpoint | None

    if checkpoint["model_type"] == "autoencoder_collection":
        cnn = checkpoint["autoencoders"][channel_index]
    else:
        part = checkpoint["submodels"][channel_index]
        mlp = part["mlp"]

        mlp_layers: tuple[int, ...] = (
            mlp["num_input_neurons"],
            *mlp["hidden_layers"],
            math.prod(mlp["output_dims"]),
        )
        mlp_description: str = "–".join(str(value) for value in mlp_layers)

        lines.append(f"MLP: {mlp_description}")
        cnn = part["cnn_autoencoder"]

    if cnn is not None:
        cnn_layers: tuple[int, ...] = (
            cnn["input_channels"],
            *cnn["features"],
        )
        cnn_description: str = "–".join(str(value) for value in cnn_layers)

        lines.append(f"CNN: {cnn_description}")
        lines.append(f"Latent: {cnn['latent_dim']}; Kernel: {cnn['kernel_size']}")

    return "\n".join(lines)


def _draw_report_panel(
    fig: Figure,
    panel_spec: SubplotSpec,
    evaluation: EvaluationBundle,
    channel_index: int,
    checkpoint: ReportCheckpoint,
) -> Axes:
    """Arrange a shared prediction plot and its channel metrics."""
    channel_name: str = evaluation["channel_names"][channel_index]

    if channel_name not in CHANNEL_LABELS:
        raise ValueError(f"No label defined for channel: {channel_name}")

    quantity_name: str = CHANNEL_LABELS[channel_name][0]

    panel_grid = panel_spec.subgridspec(
        3,
        1,
        height_ratios=(4.0, 1.6, 1.2),
        hspace=0.0,
    )

    ax: Axes = fig.add_subplot(panel_grid[0])
    metrics_ax: Axes = fig.add_subplot(panel_grid[1])
    metrics_ax.set_axis_off()
    architecture_ax: Axes = fig.add_subplot(panel_grid[2])
    architecture_ax.set_axis_off()

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

    architecture_ax.text(
        0.0,
        0.5,
        _channel_architecture_text(
            checkpoint,
            channel_index,
        ),
        transform=architecture_ax.transAxes,
        fontsize=9,
        horizontalalignment="left",
        verticalalignment="center",
        linespacing=1.1,
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


def _format_training_duration(value: float | None) -> str:
    """Format the summed training duration of the included networks."""
    duration: float | None = checked_training_duration(value)

    if duration is None:
        return "unbekannt / unvollständig"

    if duration < 60:
        return f"{duration:.2f} s"

    if duration < 3600:
        return f"{duration / 60:.2f} min"

    return f"{duration / 3600:.2f} h"


def _training_settings_lines(
    label: str,
    settings: JSONValue,
) -> list[str]:
    """Describe training settings compactly using two lines."""
    if settings is None:
        return [f"{label}: Trainingssettings nicht gespeichert"]

    if not isinstance(settings, dict):
        raise TypeError("Training settings must contain a dictionary.")

    learning_rate: str = _format_training_setting(settings.get("learning_rate"))
    weight_decay: str = _format_training_setting(settings.get("weight_decay"))
    maximum_epochs: str = _format_training_setting(settings.get("num_epochs"))
    patience: str = _format_training_setting(settings.get("patience"))

    return [
        f"{label} lr={learning_rate}; wd={weight_decay}",
        f"{label} Epochen≤{maximum_epochs}; Patience={patience}",
    ]


def _build_model_information(
    checkpoint: ReportCheckpoint,
) -> str:
    """Describe training settings and the summed network training time."""
    lines: list[str]
    total_duration: float | None
    activation: str

    if checkpoint["model_type"] == "autoencoder_collection":
        cnn_count: int = len(checkpoint["autoencoders"])
        activation = checkpoint["autoencoders"][0]["activation"]

        lines = [
            f"Modell: {cnn_count} CNNs",
            f"CNN-Seed: {checkpoint['seed']}",
        ]
        lines.extend(
            _training_settings_lines(
                "CNN",
                checkpoint["metrics"].get("training_config"),
            )
        )

        durations: list[float | None] = [
            checked_training_duration(cnn["training_duration_seconds"])
            for cnn in checkpoint["autoencoders"]
        ]

        total_duration = None

        if all(duration is not None for duration in durations):
            total_duration = checked_training_duration(
                sum(duration for duration in durations if duration is not None)
            )

    else:
        mlp_count: int = len(checkpoint["submodels"])
        included_cnn_count: int = sum(
            part["cnn_autoencoder"] is not None for part in checkpoint["submodels"]
        )
        activation = checkpoint["submodels"][0]["mlp"]["activation"]

        lines = [
            f"Modell: {mlp_count} MLPs, {included_cnn_count} CNNs",
            f"MLP-Seed: {checkpoint['seed']}",
        ]
        lines.extend(
            _training_settings_lines(
                "MLP",
                checkpoint["metrics"].get("training_config"),
            )
        )

        if included_cnn_count:
            cnn_seed: str = _format_training_setting(
                checkpoint["metrics"].get("autoencoder_seed")
            )
            lines.append(f"CNN-Seed: {cnn_seed}")
            lines.extend(
                _training_settings_lines(
                    "CNN",
                    checkpoint["metrics"].get("autoencoder_training_config"),
                )
            )

        total_duration = checkpoint["training_duration_seconds"]

    lines.insert(1, f"Aktivierung: {activation}")
    lines.append(f"Σ Trainingszeit: {_format_training_duration(total_duration)}")

    return "\n".join(lines)


@mpl.rc_context(rc={**PREDICTION_STYLE, "savefig.bbox": None})
def create_prediction_report(model_folder: str | Path) -> Path:
    """Save a single-page PDF report containing all seven output channels."""
    findfont(
        FontProperties(family="Arial"),
        fallback_to_default=False,
    )

    output_path: Path
    evaluation: EvaluationBundle
    output_path, evaluation = _load_report_data(model_folder)
    checkpoint: ReportCheckpoint = _load_report_checkpoint(
        model_folder,
        evaluation,
    )
    model_information: str = _build_model_information(checkpoint)

    fig: Figure
    panel_specs: tuple[SubplotSpec, ...]
    legend_ax: Axes
    fig, panel_specs, legend_ax = _create_report_page(
        evaluation,
        output_path.parent.parent.name,
    )

    try:
        plot_axes: list[Axes] = [
            _draw_report_panel(
                fig,
                spec,
                evaluation,
                channel_index,
                checkpoint,
            )
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
                "Subject": str(model_folder),
            },
        )
    finally:
        plt.close(fig)

    logger.info("Prediction report saved: %s", output_path)
    return output_path
