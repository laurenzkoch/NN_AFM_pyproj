from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.config_mcad_parameters import (
    CSV_DECIMAL_PLACES,
    CSV_PARAMETERS,
)
from src.modules.data_managment.csv_io import (
    import_data_from_csv,
    write_row_in_csv,
)
from src.modules.data_managment.simulation_metadata import (
    SimulationMetadata,
    is_compatible_metadata,
    prepare_result_csv,
    read_metadata,
)

logger = logging.getLogger(__name__)

type KnownPoint = tuple[int | float, ...]
type KnownPointsBySource = dict[str, set[KnownPoint]]


class ResultCache:
    def __init__(
        self,
        csv_path: Path,
        expected_metadata: SimulationMetadata,
        cache_directory: Path,
    ) -> None:
        self.csv_path = csv_path
        self.expected_metadata = expected_metadata
        self.cache_directory = cache_directory

        self.dataframes: dict[str, pd.DataFrame] = self._get_csv_files()

        self._known_points_cache: dict[
            tuple[str, ...],
            KnownPointsBySource,
        ] = {}

    def prepare_output(self) -> None:
        prepare_result_csv(
            self.csv_path,
            self.expected_metadata,
        )

    def get_row(
        self,
        source: str,
        row_index: int,
    ) -> list[float]:
        return (
            self.dataframes[source]
            .iloc[row_index][CSV_PARAMETERS]
            .to_numpy(dtype=float)
            .tolist()
        )

    def _get_csv_files(self) -> dict[str, pd.DataFrame]:
        result = {}
        csv_paths: set[Path] = {
            path.resolve() for path in self.cache_directory.glob("*.csv")
        }

        if self.csv_path.is_file():
            csv_paths.add(self.csv_path.resolve())

        for path in sorted(csv_paths):
            try:
                document = read_metadata(path)
                if not is_compatible_metadata(document, self.expected_metadata):
                    logger.info("Skip incompatible simulation setup: %s", path.name)
                    continue
                frame = import_data_from_csv(path)
                if list(frame.columns) != CSV_PARAMETERS:
                    raise ValueError("CSV schema/order differs")
                numeric = frame[CSV_PARAMETERS].to_numpy(dtype=float)
                if not np.isfinite(numeric).all():
                    raise ValueError("CSV contains nonfinite values")
            except (OSError, ValueError, KeyError) as exc:
                logger.warning("Skip channel_result source %s: %s", path.name, exc)
                continue
            result[path.resolve().as_posix()] = frame.reset_index(drop=True)
        return result

    def _get_cached_known_points(
        self,
        parameter_list: list[str],
    ) -> KnownPointsBySource:
        schema = tuple(parameter_list)

        if schema not in self._known_points_cache:
            self._known_points_cache[schema] = self._get_all_known_points(
                parameter_list
            )

        return self._known_points_cache[schema]

    @staticmethod
    def _get_known_points(
        dataframe: pd.DataFrame,
        parameter_list: list[str],
    ) -> set[KnownPoint]:
        known_points = {
            tuple(row)
            for row in dataframe[parameter_list].itertuples(
                index=True,
                name=None,
            )
        }
        return known_points

    def _get_all_known_points(self, parameter_list: list[str]) -> KnownPointsBySource:
        known_points: KnownPointsBySource = {}
        for key_csv_path, value_dataframe in self.dataframes.items():
            known_points[key_csv_path] = self._get_known_points(
                value_dataframe, parameter_list
            )
        return known_points

    def find_known_result(
        self,
        parameters: dict[str, float],
    ) -> tuple[str, int] | None:
        schema = tuple(parameters)
        target = np.asarray(tuple(parameters.values()), dtype=float)
        if not np.isfinite(target).all():
            raise ValueError("Nonfinite simulation parameters")
        known = self._get_cached_known_points(list(schema))
        order = sorted(
            known, key=lambda key: (key != self.csv_path.resolve().as_posix(), key)
        )
        for source in order:
            for record in sorted(known[source], key=lambda row: row[0]):
                candidate = np.asarray(record[1:], dtype=float)
                if np.allclose(
                    candidate,
                    target,
                    rtol=1e-10,
                    atol=1e-9,
                ):
                    return source, int(record[0])
        return None

    def append_result(self, values_row: list[float]) -> None:
        if len(values_row) != len(CSV_PARAMETERS):
            raise ValueError("Wrong channel_result width")
        values = [
            float(f"{float(value):.{CSV_DECIMAL_PLACES}f}") for value in values_row
        ]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite simulation channel_result")
        write_row_in_csv(self.csv_path, values)
        key = self.csv_path.resolve().as_posix()
        current = self.dataframes.get(key)
        row_index = 0 if current is None else len(current)
        added = pd.DataFrame([values], columns=CSV_PARAMETERS)
        self.dataframes[key] = (
            added
            if current is None or current.empty
            else pd.concat([current, added], ignore_index=True)
        )
        row = dict(zip(CSV_PARAMETERS, values, strict=True))
        for schema, sources in self._known_points_cache.items():
            record = (row_index, *(row[name] for name in schema))
            sources.setdefault(key, set()).add(record)
