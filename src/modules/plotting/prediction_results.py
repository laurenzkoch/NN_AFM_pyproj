import logging
from pathlib import Path
from typing import cast

import matplotlib as mpl
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.layout_engine import ConstrainedLayoutEngine

from src.config.config_paths import PATH_DATA_TRAINED_NN
from src.modules.neuronal_networks.evaluation_io import EvaluationBundle
from src.modules.plotting.prediction_base import (
    CHANNEL_LABELS,
    PREDICTION_STYLE,
    draw_channel_metrics,
    draw_prediction_axis,
)

logger = logging.getLogger(__name__)


@mpl.rc_context(rc=PREDICTION_STYLE)
def plot_prediction_results(model_folder: str) -> Path:
    """Save individual prediction plots using the shared drawing functions."""
    findfont(
        FontProperties(family="Arial"),
        fallback_to_default=False,
    )

    if (
        not model_folder
        or model_folder in {".", ".."}
        or Path(model_folder).name != model_folder
    ):
        raise ValueError("Please state only the model folder name.")

    model_directory: Path = Path(PATH_DATA_TRAINED_NN) / model_folder
    evaluation_path: Path = model_directory / "evaluation.pt"
    output_directory: Path = model_directory / "result_plots"

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

    output_directory.mkdir(parents=True, exist_ok=True)

    for channel_index, channel_name in enumerate(evaluation["channel_names"]):
        if channel_name not in CHANNEL_LABELS:
            raise ValueError(f"No plot label defined for channel: {channel_name}")

        quantity_name: str = CHANNEL_LABELS[channel_name][0]

        fig: Figure = plt.figure(
            layout=ConstrainedLayoutEngine(h_pad=0.20),
        )

        try:
            grid = fig.add_gridspec(
                1,
                2,
                width_ratios=(1.0, 0.85),
            )

            ax: Axes = fig.add_subplot(grid[0, 0])
            info_ax: Axes = fig.add_subplot(grid[0, 1])
            info_ax.set_axis_off()

            fig.suptitle(
                quantity_name,
                fontsize=11,
                fontweight="bold",
            )

            draw_prediction_axis(
                ax,
                evaluation,
                channel_index,
                include_counts=True,
            )

            handles, legend_labels = ax.get_legend_handles_labels()

            info_ax.legend(
                handles,
                legend_labels,
                loc="upper left",
                bbox_to_anchor=(0.0, 1.0),
                borderaxespad=0.0,
                frameon=False,
                fontsize=9,
                labelspacing=0.9,
                handlelength=1.6,
                markerscale=1.4,
            )

            info_ax.text(
                0.0,
                0.50,
                "Kennwerte",
                transform=info_ax.transAxes,
                fontsize=10,
                fontweight="bold",
            )

            draw_channel_metrics(
                info_ax,
                evaluation,
                channel_index,
                include_mean=True,
                bbox=(0.0, 0.18, 1.0, 0.27),
            )

            filename_stem: str = f"{channel_name}_prediction_vs_sim_ref-model"

            for extension in ("png", "pdf"):
                fig.savefig(
                    output_directory / f"{filename_stem}.{extension}",
                    dpi=300,
                    facecolor="white",
                )
        finally:
            plt.close(fig)

    logger.info("Prediction plots saved: %s", output_directory)
    return output_directory
