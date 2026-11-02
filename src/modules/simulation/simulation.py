from __future__ import annotations

import logging
import math
from collections.abc import Generator, Iterable
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    wait,
)
from dataclasses import dataclass, field
from itertools import batched
from multiprocessing.queues import Queue as ProcessQueue
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from src.config.config_mcad_parameters import (
    DEFAULT_VALUES,
    MCAD_RESULT_PARAMETERS,
    REFERENCE_CSV_PARAMETERS,
)
from src.config.config_paths import PATH_DATA_MCAD_RESULTS
from src.modules.data_managment.simulation_metadata import (
    build_simulation_metadata,
)
from src.modules.domain.reference_run import ReferenceRunResult
from src.modules.process.motor_cad_processes import mcad_instance_handling
from src.modules.process.motor_cad_processes.mcad_errors import (
    MCADResultError,
)
from src.modules.runtime.cancellation import CancellationToken
from src.modules.runtime.events import EventType, PipelineEvent
from src.modules.simulation import workers
from src.modules.simulation.result_cache import ResultCache

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    parameters: dict[str, float]
    reference: ReferenceRunResult
    base_speed_rpm: float
    dc_bus_voltage_v: float
    shaft_speed_norm: float
    log_context: str = ""

    def csv_reference_values(self) -> dict[str, float]:
        return {
            "ReferenceSpeedRPM": self.reference.reference_speed_rpm,
            "ReferenceKeFundamental": (
                self.reference.back_emf_constant_ll_peak_vs_per_rad
            ),
            "ReferencePeakBackEMFPhase": self.reference.peak_back_emf_phase_v,
            "BaseSpeedRPM": self.base_speed_rpm,
            "ShaftSpeedNorm": self.shaft_speed_norm,
        }


@dataclass
class PendingRequest:
    schema: tuple[str, ...]
    point: NDArray[np.float64]
    future: Future[tuple[list[float] | None, bool]]
    operating_point: OperatingPoint
    count: int = 1
    log_context: str = ""
    submitted_at: float = field(default_factory=perf_counter)


class MCADSimulation:
    def __init__(
        self,
        geometry_generator: Generator[dict[str, float]],
        operating_map: NDArray[np.float64],
        batch_size: int,
        progress_queue: ProcessQueue[PipelineEvent],
        log_queue: ProcessQueue[logging.LogRecord],
        csv_path: Path,
        num_workers: int,
        temp_root: Path | None = None,
        cancellation: CancellationToken | None = None,
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

        if type(num_workers) is not int or num_workers < 1:
            raise ValueError("num_workers must be a positive integer.")

        validated_map = np.array(
            operating_map,
            dtype=np.float64,
            copy=True,
        )

        if (
            validated_map.ndim != 2
            or validated_map.shape[1] != 3
            or validated_map.shape[0] == 0
        ):
            raise ValueError(
                "operating_map must have shape (N, 3) with N greater than zero."
            )

        if not np.isfinite(validated_map).all():
            raise ValueError("operating_map must contain only finite values.")

        if (validated_map[:, 0] < 0).any():
            raise ValueError("Normalized speed must not be negative.")

        validated_map.setflags(write=False)

        points_per_geometry = len(validated_map)

        if batch_size % points_per_geometry != 0:
            raise ValueError("batch_size must be a multiple of the operating-map size.")

        self.geometry_generator = geometry_generator
        self.operating_map: NDArray[np.float64] = validated_map
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

        self.result_cache = ResultCache(
            csv_path=self.csv_path,
            expected_metadata=self.expected_metadata,
            cache_directory=PATH_DATA_MCAD_RESULTS,
        )

    def prepare_output(self) -> None:
        self.result_cache.prepare_output()

    def run(self) -> None:
        self.cancellation.raise_if_requested()

        dc_bus_voltage_v = float(DEFAULT_VALUES["DCBusVoltage"])

        if not math.isfinite(dc_bus_voltage_v) or dc_bus_voltage_v <= 0:
            raise ValueError("DCBusVoltage must be finite and positive.")

        geometries_per_pool = self.batch_size // len(self.operating_map)

        for geometry_batch in batched(
            self.geometry_generator,
            geometries_per_pool,
        ):
            self.cancellation.raise_if_requested()

            self._run_geometry_batch(
                geometry_batch,
                dc_bus_voltage_v,
            )

        self.cancellation.raise_if_requested()

    def _run_geometry_batch(
        self,
        geometries: Iterable[dict[str, float]],
        dc_bus_voltage_v: float,
    ) -> None:
        self.cancellation.raise_if_requested()

        batch_context = f"batch={uuid4().hex[:12]}"
        started = perf_counter()
        phase = "pool_creation"

        logger.info(
            "Process pool creation started: "
            "%s workers=%d planned_load_point_limit=%d csv=%s",
            batch_context,
            self.num_workers,
            self.batch_size,
            self.csv_path,
        )

        try:
            executor = ProcessPoolExecutor(
                max_workers=self.num_workers,
                initializer=mcad_instance_handling.init_mcad_instance,
                initargs=(self.log_queue, self.temp_root),
            )
        except BaseException:
            logger.error(
                "Process pool creation failed: %s",
                batch_context,
            )
            raise

        logger.info(
            "Process pool created; workers initialize on demand: %s",
            batch_context,
        )

        geometry_count = 0

        try:
            for geometry_count, geometry in enumerate(
                geometries,
                start=1,
            ):
                self.cancellation.raise_if_requested()

                geometry_context = f"{batch_context} geometry={uuid4().hex[:12]}"
                reference_context = f"{geometry_context} task={uuid4().hex[:12]}"

                logger.debug(
                    "Geometry parameters: %s parameters=%r",
                    geometry_context,
                    geometry,
                )

                reference_parameters = {
                    **geometry,
                    "DCBusVoltage": dc_bus_voltage_v,
                }

                phase = "reference_submission"
                logger.info(
                    "Submitting reference task: %s",
                    reference_context,
                )

                reference_future = executor.submit(
                    workers.execute_reference_simulation,
                    reference_parameters,
                    log_context=reference_context,
                )

                phase = "reference_wait"
                reference = self._wait_for_reference(reference_future)

                if reference is None:
                    dropped_points = len(self.operating_map)

                    logger.warning(
                        "Reference rejected; dropping operating map and "
                        "continuing with next geometry: "
                        "%s dropped_points=%d reference_parameters=%r",
                        reference_context,
                        dropped_points,
                        reference_parameters,
                    )

                    self._report_progress(
                        "dropped",
                        dropped_points,
                    )
                    continue

                phase = "operating_point_conversion"
                base_speed_rpm = reference.calculate_base_speed_rpm(dc_bus_voltage_v)

                logger.info(
                    "Reference completed: "
                    "%s reference_speed_rpm=%.6f "
                    "Ke=%.12g peak_phase_back_emf_v=%.6f "
                    "dc_bus_voltage_v=%.6f base_speed_rpm=%.6f",
                    reference_context,
                    reference.reference_speed_rpm,
                    reference.back_emf_constant_ll_peak_vs_per_rad,
                    reference.peak_back_emf_phase_v,
                    dc_bus_voltage_v,
                    base_speed_rpm,
                )

                operating_points = self._generate_operating_points(
                    geometry,
                    reference,
                    dc_bus_voltage_v,
                    log_context=geometry_context,
                )

                phase = "load_processing"
                self._run_simulation_batch(
                    executor,
                    operating_points,
                )

            self.cancellation.raise_if_requested()

        except BaseException as exc:
            logger.error(
                "Geometry batch aborted: "
                "%s phase=%s elapsed_s=%.3f "
                "error_type=%s detail=%s",
                batch_context,
                phase,
                perf_counter() - started,
                type(exc).__name__,
                exc,
            )

            self._stop_pool_after_error(
                executor,
                batch_context,
            )
            raise

        else:
            logger.info(
                "Closing process pool: %s",
                batch_context,
            )

            try:
                executor.shutdown(wait=True)
            except BaseException:
                logger.error(
                    "Process pool shutdown failed: %s",
                    batch_context,
                )
                self._stop_pool_after_error(
                    executor,
                    batch_context,
                )
                raise

            logger.info(
                "Geometry batch completed and pool shut down: "
                "%s geometries=%d elapsed_s=%.3f",
                batch_context,
                geometry_count,
                perf_counter() - started,
            )

    def _generate_operating_points(
        self,
        geometry: dict[str, float],
        reference: ReferenceRunResult,
        dc_bus_voltage_v: float,
        *,
        log_context: str = "",
    ) -> Generator[OperatingPoint]:
        base_speed_rpm = reference.calculate_base_speed_rpm(dc_bus_voltage_v)

        for speed_norm, q_density, d_density in self.operating_map:
            self.cancellation.raise_if_requested()

            shaft_speed_norm = float(speed_norm)
            q_current_density = float(q_density)
            d_current_density = float(d_density)

            shaft_speed_rpm = shaft_speed_norm * base_speed_rpm
            current_density = math.hypot(
                q_current_density,
                d_current_density,
            )
            phase_advance = math.degrees(
                math.atan2(-d_current_density, q_current_density)
            )

            if not all(
                math.isfinite(value)
                for value in (
                    shaft_speed_rpm,
                    current_density,
                    phase_advance,
                )
            ):
                raise ValueError(
                    "Operating-point conversion produced nonfinite values."
                )

            parameters = {
                **geometry,
                "DCBusVoltage": dc_bus_voltage_v,
                "ShaftSpeed": shaft_speed_rpm,
                "RMSCurrentDensity": current_density,
                "PhaseAdvance": phase_advance,
            }

            yield OperatingPoint(
                parameters=parameters,
                reference=reference,
                base_speed_rpm=base_speed_rpm,
                dc_bus_voltage_v=dc_bus_voltage_v,
                shaft_speed_norm=shaft_speed_norm,
                log_context=log_context,
            )

    def _wait_for_reference(
        self,
        future: Future[ReferenceRunResult | None],
    ) -> ReferenceRunResult | None:
        while True:
            self.cancellation.raise_if_requested()

            completed, _ = wait(
                [future],
                timeout=0.1,
                return_when=FIRST_COMPLETED,
            )

            self.cancellation.raise_if_requested()

            if completed:
                return future.result()

    def _run_simulation_batch(
        self,
        executor: ProcessPoolExecutor,
        batch: Iterable[OperatingPoint],
    ) -> None:
        self.cancellation.raise_if_requested()

        requests: list[PendingRequest] = []

        self._submit_batch(
            executor,
            batch,
            requests,
        )
        self._collect_results(requests)

        self.cancellation.raise_if_requested()

    def _submit_batch(
        self,
        executor: ProcessPoolExecutor,
        batch: Iterable[OperatingPoint],
        requests: list[PendingRequest],
    ) -> None:
        for operating_point in batch:
            self.cancellation.raise_if_requested()

            parameters = operating_point.parameters
            lookup_parameters = {
                **parameters,
                **operating_point.csv_reference_values(),
            }

            known = self.result_cache.find_known_result(lookup_parameters)

            if known is not None:
                source, row_index = known

                logger.debug(
                    "Reusing existing result: %s source=%s row_index=%d",
                    operating_point.log_context,
                    source,
                    row_index,
                )

                if source == self.csv_path.resolve().as_posix():
                    self._report_progress("existing")
                else:
                    row = self.result_cache.get_row(
                        source,
                        row_index,
                    )
                    self.result_cache.append_result(row)
                    self._report_progress("copied")

                continue

            schema = tuple(lookup_parameters)
            point = np.asarray(
                tuple(lookup_parameters.values()),
                dtype=np.float64,
            )

            duplicate = next(
                (
                    request
                    for request in requests
                    if request.schema == schema
                    and np.allclose(
                        request.point,
                        point,
                        rtol=1e-10,
                        atol=1e-9,
                    )
                ),
                None,
            )

            if duplicate is not None:
                duplicate.count += 1
                logger.debug(
                    "Combining duplicate request: %s original_task=%s count=%d",
                    operating_point.log_context,
                    duplicate.log_context,
                    duplicate.count,
                )
                continue

            task_context = (
                f"{operating_point.log_context} task={uuid4().hex[:12]}"
            ).strip()

            logger.debug(
                "Submitting load task: %s",
                task_context,
            )

            future = executor.submit(
                workers.execute_simulation,
                parameters,
                log_context=task_context,
            )

            requests.append(
                PendingRequest(
                    schema=schema,
                    point=point,
                    future=future,
                    operating_point=operating_point,
                    log_context=task_context,
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
                try:
                    row, succeeded = future.result()
                except BaseException as exc:
                    logger.error(
                        "Load task result failed: "
                        "%s turnaround_s=%.3f error_type=%s detail=%s",
                        request.log_context,
                        perf_counter() - request.submitted_at,
                        type(exc).__name__,
                        exc,
                    )
                    raise

                if not succeeded:
                    self._report_progress("dropped", request.count)
                    logger.warning(
                        "Load task rejected: %s count=%d",
                        request.log_context,
                        request.count,
                    )
                    continue

                if row is None:
                    raise MCADResultError(
                        "Successful load task returned no results: "
                        f"{request.log_context}"
                    )

                self.cancellation.raise_if_requested()
                if len(row) != len(MCAD_RESULT_PARAMETERS):
                    raise MCADResultError(
                        "Wrong Motor-CAD result width: "
                        f"{request.log_context}; "
                        f"expected={len(MCAD_RESULT_PARAMETERS)}, "
                        f"actual={len(row)}"
                    )

                reference_values = request.operating_point.csv_reference_values()

                csv_row = [
                    *row,
                    *(reference_values[name] for name in REFERENCE_CSV_PARAMETERS),
                ]

                try:
                    self.result_cache.append_result(csv_row)
                except Exception:
                    logger.exception(
                        "Could not store load result: %s csv=%s",
                        request.log_context,
                        self.csv_path,
                    )
                    raise

                logger.debug(
                    "Load result stored: %s turnaround_s=%.3f count=%d csv=%s",
                    request.log_context,
                    perf_counter() - request.submitted_at,
                    request.count,
                    self.csv_path,
                )

                self._report_progress("simulated", request.count)

    @staticmethod
    def _stop_pool_after_error(
        executor: ProcessPoolExecutor,
        log_context: str,
    ) -> None:
        started = perf_counter()

        logger.warning(
            "Stopping process pool after error: %s",
            log_context,
        )

        try:
            executor.kill_workers()
        except (
            OSError,
            RuntimeError,
        ) as cleanup_error:
            logger.exception(
                "Process pool termination failed: %s detail=%s",
                log_context,
                cleanup_error,
            )
        else:
            logger.info(
                "Process pool termination completed: %s elapsed_s=%.3f",
                log_context,
                perf_counter() - started,
            )

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
