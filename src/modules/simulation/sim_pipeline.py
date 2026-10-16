from __future__ import annotations

import logging
import math
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from multiprocessing.queues import Queue as ProcessQueue
from pathlib import Path
from typing import Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

from src.config.config_mcad_parameters import MAP_SIZE
from src.config.config_paths import (
    PATH_DATA_MCAD_RESULTS,
    PATH_MCAD_DATA,
    PATH_TEMP_FILES,
)
from src.modules.data_managment.simulation_metadata import (
    build_simulation_metadata,
    prepare_result_csv,
)
from src.modules.Pipeline_and_GUI.events import EventType, PipelineEvent
from src.modules.Pipeline_and_GUI.pipeline import MyPipeline
from src.modules.process.eduvpn import EDUVPNProcessor
from src.modules.simulation.generators.machine_generator import AFMMachineGenerator
from src.modules.simulation.simulation import MCADSimulation

GeneratorType = Literal["v1", "v2"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulationJob:
    num_base_designs: int
    num_lam_designs: int
    csv_path: Path
    point_count: int


class SimulationSeries:
    def __init__(
        self,
        jobs: tuple[SimulationJob, ...],
        gen_type: str,
        num_workers: int,
        progress_queue: ProcessQueue[PipelineEvent],
        log_queue: ProcessQueue[logging.LogRecord],
    ) -> None:
        self.jobs: tuple[SimulationJob, ...] = jobs
        self.gen_type: str = gen_type
        self.num_workers: int = num_workers

        self.progress_queue: ProcessQueue[PipelineEvent] = progress_queue
        self.log_queue: ProcessQueue[logging.LogRecord] = log_queue

        self._prepared: bool = False
        self.temp_root: Path | None = None

    def prepare_output(self) -> None:
        self._prepared = False
        metadata = build_simulation_metadata()

        for job in self.jobs:
            prepare_result_csv(job.csv_path, metadata)

        self._prepared = True

    def run(self) -> None:
        if not self._prepared:
            raise RuntimeError("The results files have not been prepared.")

        for index, job in enumerate(self.jobs, start=1):
            self.progress_queue.put(
                PipelineEvent(
                    event_type=EventType.SIMULATION_STARTED,
                    text=f"Files {index} from {len(self.jobs)}",
                    max_steps=job.point_count,
                    combination_index=index,
                )
            )

            generator = AFMMachineGenerator(
                num_base_designs=job.num_base_designs,
                num_lam_designs=job.num_lam_designs,
                num_data_points=job.point_count,
            )

            workers: int = min(self.num_workers, job.point_count)

            simulation = MCADSimulation(
                point_generator=generator.get_generator(_type=self.gen_type),
                batch_size=workers * math.prod(MAP_SIZE),
                progress_queue=self.progress_queue,
                log_queue=self.log_queue,
                csv_path=job.csv_path,
                num_workers=workers,
                temp_root=self.temp_root,
            )

            simulation.prepare_output()
            simulation.run()

            self.progress_queue.put(
                PipelineEvent(
                    event_type=EventType.SIMULATION_FINISHED,
                    text=f"Kombination {index} abgeschlossen",
                    combination_index=index,
                )
            )


class SimulationPipeline(MyPipeline):
    def __init__(
        self,
        design_combinations: Sequence[tuple[int, int]],
        start_eduvpn: bool = True,
        gen_type: str = "v1",
        csv_paths: Sequence[str | Path] | None = None,
        num_workers: int = 4,
    ) -> None:
        combinations = self._validate_combinations(
            design_combinations,
            gen_type,
        )

        if type(num_workers) is not int or num_workers < 1:
            raise ValueError("num_workers muss eine positive Ganzzahl sein.")

        jobs = self._build_jobs(
            combinations=combinations,
            gen_type=gen_type,
            csv_paths=csv_paths,
            run_id=uuid4().hex[:8],
        )

        super().__init__(
            service_name="MCAD Simulation",
            num_kernel=num_workers,
        )

        self.jobs: tuple[SimulationJob, ...] = jobs
        self.gen_type: str = gen_type
        self.start_eduvpn: bool = start_eduvpn
        self.eduvpn_processor: EDUVPNProcessor | None = None

        self.status_text: str = "Ready"
        self.simulated_points: int = 0
        self.dropped_points: int = 0
        self.existing_points: int = 0
        self.copied_points: int = 0

        self.max_progress_steps = jobs[0].point_count
        self.overall_max_value = len(jobs)
        self.overall_value = 0

        self.service: SimulationSeries = SimulationSeries(
            jobs=self.jobs,
            gen_type=self.gen_type,
            num_workers=self.num_workers,
            progress_queue=self.progress_queue,
            log_queue=self.log_queue,
        )

    def run(self) -> None:
        self.start()

    def start(self) -> None:
        self._claim_run()

        run_directory: Path | None = None

        try:
            self.service.prepare_output()

            temp_parent: Path = PATH_MCAD_DATA / PATH_TEMP_FILES
            temp_parent.mkdir(parents=True, exist_ok=True)

            run_directory = Path(
                tempfile.mkdtemp(
                    dir=temp_parent,
                    prefix="simulation_run_",
                )
            )
            self.service.temp_root = run_directory

            if self.start_eduvpn:
                self.eduvpn_processor = EDUVPNProcessor()
                self.eduvpn_processor.start()

            self._run_pipeline()

        finally:
            self.service.temp_root = None

            if run_directory is not None:
                try:
                    shutil.rmtree(run_directory)
                except OSError:
                    logger.warning(
                        "Could not completely remove temporary directory: %s",
                        run_directory,
                        exc_info=True,
                    )

            if self.eduvpn_processor is not None:
                try:
                    self.eduvpn_processor.stop()
                except Exception:
                    logger.warning(
                        "Could not stop eduVPN",
                        exc_info=True,
                    )
            self._close_queues()

    @staticmethod
    def _count_points(
        num_base_designs: int,
        num_lam_designs: int,
        gen_type: str,
    ) -> int:
        counts = (num_base_designs, num_lam_designs)

        if any(type(value) is not int or value < 1 for value in counts):
            raise ValueError("Design numbers must be positive integers.")

        match gen_type:
            case "v1":
                designs = num_base_designs * num_lam_designs**3
            case "v2":
                designs = num_base_designs * num_lam_designs
            case _:
                raise ValueError(f"Unknown generator type: {gen_type}")

        return designs * math.prod(MAP_SIZE)

    @staticmethod
    def _create_job_csv_path(
        num_base_designs: int,
        num_lam_designs: int,
        gen_type: str,
        run_id: str,
    ) -> Path:
        timestamp = datetime.now(ZoneInfo("Europe/Berlin")).strftime(
            "%d.%m.%Y_%H-%M-%S_%Z"
        )

        filename = (
            f"Results_EMagnetic_simulation_"
            f"{num_base_designs}_"
            f"{num_lam_designs}_"
            f"{MAP_SIZE[0]}_"
            f"GenType-{gen_type}_"
            f"{timestamp}_{run_id}.csv"
        )

        return PATH_DATA_MCAD_RESULTS / filename

    def _validate_combinations(
        self,
        design_combinations: Sequence[tuple[int, int]],
        gen_type: str,
    ) -> tuple[tuple[int, int], ...]:
        combinations = tuple(design_combinations)

        if not combinations:
            raise ValueError("At least one combination is required.")

        available_pairs = len(AFMMachineGenerator.find_pole_slot_pairs())

        for pair in combinations:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("Each combination must be a tuple (B, L).")

            self._count_points(pair[0], pair[1], gen_type)

            if pair[0] > available_pairs:
                raise ValueError(
                    f"A maximum of {available_pairs} basic designs are possible."
                )

        if len(set(combinations)) != len(combinations):
            raise ValueError("Duplicate combinations are not permitted.")

        return combinations

    def _build_jobs(
        self,
        combinations: tuple[tuple[int, int], ...],
        gen_type: str,
        csv_paths: Sequence[str | Path] | None,
        run_id: str,
    ) -> tuple[SimulationJob, ...]:
        if csv_paths is not None and len(csv_paths) != len(combinations):
            raise ValueError("Pro Kombination ist genau ein CSV-Pfad nötig.")

        jobs: list[SimulationJob] = []

        for index, (base_designs, lam_designs) in enumerate(combinations):
            if csv_paths is None:
                path = self._create_job_csv_path(
                    base_designs,
                    lam_designs,
                    gen_type,
                    run_id,
                )
            else:
                path = Path(csv_paths[index])

                if not path.is_absolute():
                    path = PATH_DATA_MCAD_RESULTS / path

            if path.suffix.lower() != ".csv":
                raise ValueError("Ergebnisdateien müssen auf .csv enden.")

            jobs.append(
                SimulationJob(
                    num_base_designs=base_designs,
                    num_lam_designs=lam_designs,
                    csv_path=path,
                    point_count=self._count_points(
                        base_designs,
                        lam_designs,
                        gen_type,
                    ),
                )
            )

        paths = {job.csv_path.resolve() for job in jobs}

        if len(paths) != len(jobs):
            raise ValueError("Die CSV-Zielpfade müssen unterschiedlich sein.")

        return tuple(jobs)

    def _queue_adapter(
        self,
        event: PipelineEvent,
    ) -> tuple[int, str, bool]:
        finished = False

        if event.event_type == EventType.STARTED:
            self.status_text = event.text

        elif event.event_type == EventType.SIMULATION_STARTED:
            if event.max_steps is None or event.max_steps < 1:
                raise ValueError("Invalid simulation score.")

            self.max_progress_steps = event.max_steps
            self.simulated_points = 0
            self.dropped_points = 0
            self.existing_points = 0
            self.copied_points = 0
            self.status_text = event.text

        elif event.event_type == EventType.SIMULATION_FINISHED:
            if (
                event.combination_index is None
                or not 1 <= event.combination_index <= len(self.jobs)
            ):
                raise ValueError("Invalid combination number.")

            self.overall_value = event.combination_index
            self.status_text = event.text

        elif event.event_type == EventType.PROGRESS:
            if event.outcome == "simulated":
                self.simulated_points += event.increment
            elif event.outcome == "dropped":
                self.dropped_points += event.increment
            elif event.outcome == "existing":
                self.existing_points += event.increment
            elif event.outcome == "copied":
                self.copied_points += event.increment

        elif event.event_type == EventType.FINISHED:
            self.status_text = event.text
            finished = True

        elif event.event_type == EventType.FAILED:
            self.status_text = event.text
            finished = True
        value = (
            self.simulated_points
            + self.dropped_points
            + self.existing_points
            + self.copied_points
        )
        return value, self.status_text, finished
