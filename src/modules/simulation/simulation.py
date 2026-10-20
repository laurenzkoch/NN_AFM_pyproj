from __future__ import annotations

import logging
from collections.abc import Generator, Iterable
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    wait,
)
from dataclasses import dataclass
from itertools import batched
from multiprocessing.queues import Queue as ProcessQueue
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config.config_mcad_parameters import CSV_PARAMETERS
from src.config.config_paths import PATH_DATA_MCAD_RESULTS
from src.modules.data_managment.csv_io import import_data_from_csv, write_row_in_csv
from src.modules.data_managment.simulation_metadata import (
    build_simulation_metadata,
    is_compatible_metadata,
    prepare_result_csv,
    read_metadata,
)
from src.modules.Pipeline_and_GUI.cancellation import (
    CancellationToken,
    PipelineCancelledError,
)
from src.modules.Pipeline_and_GUI.events import EventType, PipelineEvent
from src.modules.process.motor_cad_processes import mcad_instance_handling

logger = logging.getLogger(__name__)

type KnownPoint = tuple[int | float, ...]
type KnownPointsBySource = dict[str, set[KnownPoint]]


@dataclass
class PendingRequest:
    schema: tuple[str, ...]
    point: NDArray[np.float64]
    future: Future[tuple[list[float] | None, bool]]
    count: int = 1


def _execute_simulation(
    parameter_dict: dict[str, float],
) -> tuple[list[float] | None, bool]:
    try:
        mcad = mcad_instance_handling.get_mcad_instance()

        for name, value in parameter_dict.items():
            # np.float64, np.int64 etc. -> Python float/int
            if isinstance(value, np.generic):
                value = value.item()

            if name == "Pole_Number":
                tppc = mcad.get_value_from_mcad("TorquePointsPerCycle")
                mesh_points = value / 2 * tppc
                if mesh_points < 360:
                    mesh_points = 4 * mesh_points
                mcad.set_value_in_mcad(
                    variable_name="AirgapMeshPoints_mesh", value=mesh_points
                )
                mcad.set_value_in_mcad(
                    variable_name="AirgapMeshPoints_layers", value=mesh_points
                )

            mcad.set_value_in_mcad(name, value)

        return mcad.execute_mcad_simulation()

    except BaseException:
        try:
            mcad_instance_handling.shutdown_mcad_instance()
        except Exception:
            logger.exception("MCAD cleanup failed after simulation error")
        raise


class MCADSimulation:
    def __init__(
        self,
        point_generator: Generator[dict[str, float]],
        batch_size: int,
        progress_queue: ProcessQueue[PipelineEvent],
        log_queue: ProcessQueue[logging.LogRecord],
        csv_path: Path,
        num_workers: int,
        temp_root: Path | None = None,
        cancellation: CancellationToken | None = None,
    ) -> None:
        self.point_generator = point_generator
        self.batch_size = batch_size
        self.progress_queue: ProcessQueue[PipelineEvent] = progress_queue
        self.log_queue: ProcessQueue[logging.LogRecord] = log_queue
        self.csv_path = csv_path
        self.num_workers = num_workers
        self.temp_root: Path | None = temp_root
        self.cancellation: CancellationToken = (
            cancellation if cancellation is not None else CancellationToken()
        )

        self.expected_metadata = build_simulation_metadata()
        self.dataframes: dict[str, pd.DataFrame] = self._get_csv_files()

        self._known_points_cache: dict[
            tuple[str, ...],
            KnownPointsBySource,
        ] = {}

    def prepare_output(self) -> None:
        prepare_result_csv(self.csv_path, self.expected_metadata)

    def run(self) -> None:
        try:
            self.cancellation.raise_if_requested()

            for batch in batched(self.point_generator, self.batch_size):
                self.cancellation.raise_if_requested()
                self._run_simulation_pool(batch)

            self.cancellation.raise_if_requested()

        except PipelineCancelledError:
            logger.info("Simulation cancelled: %s", self.csv_path.name)
            raise

        except Exception:
            logger.exception("An error occurred during simulation.")
            raise

    def _get_csv_files(self) -> dict[str, pd.DataFrame]:
        result = {}
        csv_paths: set[Path] = {
            path.resolve() for path in PATH_DATA_MCAD_RESULTS.glob("*.csv")
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
                logger.warning("Skip result source %s: %s", path.name, exc)
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

    def _find_known_result(
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
                if np.all(np.abs(candidate - target) <= 1e-4):
                    return source, int(record[0])
        return None

    def _append_result(self, values_row: list[float]) -> None:
        if len(values_row) != len(CSV_PARAMETERS):
            raise ValueError("Wrong result width")
        # Match the existing CSV writer's four-decimal storage precision.
        values = [float(f"{float(value):.4f}") for value in values_row]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite simulation result")
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

    def _report_progress(
        self,
        outcome: str,
        increment: int = 1,
    ) -> None:
        self.progress_queue.put(
            PipelineEvent(
                event_type=EventType.PROGRESS,
                increment=increment,
                outcome=outcome,
            )
        )

    def _submit_batch(
        self,
        executor: ProcessPoolExecutor,
        batch: Iterable[dict[str, float]],
        requests: list[PendingRequest],
    ) -> None:
        for parameters in batch:
            self.cancellation.raise_if_requested()
            known = self._find_known_result(parameters)

            if known is not None:
                source, row_index = known

                if source == self.csv_path.resolve().as_posix():
                    self._report_progress("existing")
                else:
                    row = (
                        self.dataframes[source]
                        .iloc[row_index][CSV_PARAMETERS]
                        .to_numpy(dtype=float)
                        .tolist()
                    )
                    self._append_result(row)
                    self._report_progress("copied")

                continue

            schema = tuple(parameters)
            point = np.asarray(tuple(parameters.values()), dtype=float)

            duplicate = next(
                (
                    request
                    for request in requests
                    if request.schema == schema
                    and np.all(np.abs(request.point - point) <= 1e-4)
                ),
                None,
            )

            if duplicate is not None:
                duplicate.count += 1
                continue

            future = executor.submit(_execute_simulation, parameters)

            requests.append(
                PendingRequest(
                    schema=schema,
                    point=point,
                    future=future,
                )
            )

    def _collect_results(self, requests: list[PendingRequest]) -> None:
        pending: dict[
            Future[tuple[list[float] | None, bool]],
            PendingRequest,
        ] = {request.future: request for request in requests}

        while pending:
            self.cancellation.raise_if_requested()

            completed, _ = wait(
                pending,
                timeout=0.1,
                return_when=FIRST_COMPLETED,
            )

            self.cancellation.raise_if_requested()

            for future in completed:
                self.cancellation.raise_if_requested()

                request: PendingRequest = pending.pop(future)
                row, succeeded = future.result()

                if not succeeded:
                    self._report_progress("dropped", request.count)
                    continue

                if row is None:
                    raise RuntimeError(
                        "The simulation was successful but did not produce any results."
                    )

                self.cancellation.raise_if_requested()
                self._append_result(row)
                self._report_progress("simulated", request.count)

    def _run_simulation_pool(
        self,
        batch_machine_generator_instance: Iterable[dict[str, float]],
    ) -> None:
        self.cancellation.raise_if_requested()

        requests: list[PendingRequest] = []

        executor: ProcessPoolExecutor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=mcad_instance_handling.init_mcad_instance,
            initargs=(self.log_queue, self.temp_root),
        )

        try:
            self._submit_batch(
                executor,
                batch_machine_generator_instance,
                requests,
            )
            self._collect_results(requests)
            self.cancellation.raise_if_requested()

        except BaseException:
            for request in requests:
                request.future.cancel()

            executor.kill_workers()
            raise

        else:
            executor.shutdown(wait=True)
