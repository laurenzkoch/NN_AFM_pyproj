"""Select optimization sources matching the training metadata."""

import logging
from pathlib import Path

from src.modules.data_managment.experiment_utils import file_sha256
from src.modules.data_managment.map_grid import MapGrid
from src.modules.data_managment.simulation_metadata import (
    is_compatible_metadata,
    read_metadata,
)

logger = logging.getLogger(__name__)


def discover_test_sources(
    directory: Path,
    training_filename: str,
    *,
    excluded_source_hash: str | None = None,
) -> tuple[str, ...]:
    if (
        Path(training_filename).name != training_filename
        or Path(training_filename).suffix.lower() != ".csv"
    ):
        raise ValueError("Use CSV filenames without a directory.")

    training_path = directory / training_filename

    if not training_path.is_file():
        raise FileNotFoundError(training_path)

    reference = read_metadata(training_path)

    if not is_compatible_metadata(reference, reference):
        raise ValueError(
            f"Training metadata needs explicit migration: {training_filename}"
        )

    grid = MapGrid.from_payload(reference["payload"].get("map_grid"))

    logger.info(
        "Training source grid: csv=%s map_grid=%r",
        training_filename,
        grid.to_payload(),
    )

    selected: list[str] = []

    for path in sorted(directory.iterdir()):
        if (
            not path.is_file()
            or path.suffix.lower() != ".csv"
            or path.resolve() == training_path.resolve()
        ):
            continue

        try:
            candidate = read_metadata(path)

            if not is_compatible_metadata(
                candidate,
                reference,
            ):
                logger.warning(
                    "Excluded optimization source: csv=%s reason=incompatible_metadata",
                    path.name,
                )
                continue

        except (OSError, ValueError) as exc:
            logger.warning(
                "Excluded optimization source: "
                "csv=%s reason=invalid_metadata detail=%s",
                path.name,
                exc,
            )
            continue

        if (
            excluded_source_hash is not None
            and file_sha256(path) == excluded_source_hash
        ):
            logger.info(
                "Excluded CNN training source from independent tests: %s",
                path.name,
            )
            continue

        selected.append(path.name)

    return tuple(selected)
