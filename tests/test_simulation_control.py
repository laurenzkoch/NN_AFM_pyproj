from __future__ import annotations

import logging
import math
from concurrent.futures import Future
from multiprocessing.queues import Queue as ProcessQueue
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from src.config.config_mcad_parameters import MAP_SIZE
from src.modules.data_managment.simulation_metadata import SimulationMetadata
from src.modules.Pipeline_and_GUI.events import EventType, PipelineEvent
from src.modules.simulation import sim_pipeline as pipelines
from src.modules.simulation import simulation as simulations
from src.modules.simulation.reference_run import ReferenceRunResult

type EventQueues = tuple[ProcessQueue[PipelineEvent], ProcessQueue[logging.LogRecord]]


def assert_queues_closed(
    progress: ProcessQueue[PipelineEvent],
    logs: ProcessQueue[logging.LogRecord],
) -> None:
    with pytest.raises(ValueError, match="closed"):
        progress.put(PipelineEvent(event_type=EventType.PROGRESS))

    with pytest.raises(ValueError, match="closed"):
        logs.put(logging.makeLogRecord({"msg": "queue closure test"}))


@pytest.mark.parametrize(
    "kind,bases,lambdas,expected", [("v1", 2, 3, 54), ("v2", 2, 3, 6)]
)
def test_point_count_formulas(
    kind: str, bases: int, lambdas: int, expected: int
) -> None:
    assert pipelines.SimulationPipeline._count_points(
        bases, lambdas, kind
    ) == expected * math.prod(MAP_SIZE)


@pytest.mark.parametrize(
    "combinations", [[], [(0, 1)], [(True, 1)], [(1, 1), (1, 1)], [[1, 1]]]
)
def test_invalid_combinations_are_rejected(combinations: object) -> None:
    with pytest.raises(ValueError):
        pipelines.SimulationPipeline(
            design_combinations=cast(list[tuple[int, int]], combinations),
            start_eduvpn=False,
        )


def test_start_failure_closes_queues_and_prevents_second_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = pipelines.SimulationPipeline(
        [(1, 1)], start_eduvpn=False, csv_paths=[tmp_path / "unused.csv"], num_workers=1
    )
    error = ValueError("original startup failure")

    def fail() -> None:
        raise error

    monkeypatch.setattr(pipeline.service, "prepare_output", fail)
    with pytest.raises(ValueError) as captured:
        pipeline.start()
    assert captured.value is error
    assert_queues_closed(
        pipeline.progress_queue,
        pipeline.log_queue,
    )
    with pytest.raises(RuntimeError, match="already been used"):
        pipeline.start()


def test_success_removes_only_owned_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata: SimulationMetadata,
) -> None:
    monkeypatch.setattr(
        "src.modules.simulation.sim_pipeline.PATH_MCAD_DATA",
        tmp_path / "motorcad",
    )
    monkeypatch.setattr(
        "src.modules.simulation.sim_pipeline.build_simulation_metadata",
        lambda: metadata,
    )

    sentinel: Path = tmp_path / "motorcad" / "keep.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("keep", encoding="utf-8")

    owned: list[Path] = []

    class HeadlessPipeline(pipelines.SimulationPipeline):
        def _run_pipeline(self) -> None:
            assert self.service.temp_root is not None
            owned.append(self.service.temp_root)
            assert owned[0].is_dir()

    pipeline: HeadlessPipeline = HeadlessPipeline(
        [(1, 1)],
        start_eduvpn=False,
        csv_paths=[tmp_path / "output.csv"],
        num_workers=1,
    )

    pipeline.start()

    assert len(owned) == 1
    assert not owned[0].exists()
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert pipeline.service.temp_root is None

    assert_queues_closed(
        pipeline.progress_queue,
        pipeline.log_queue,
    )


def test_combinations_run_sequentially_and_reset_both_progress_levels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metadata: SimulationMetadata
) -> None:
    monkeypatch.setattr(
        "src.modules.simulation.sim_pipeline.build_simulation_metadata",
        lambda: metadata,
    )
    monkeypatch.setattr(
        "src.modules.simulation.simulation.build_simulation_metadata",
        lambda: metadata,
    )
    monkeypatch.setattr(
        "src.modules.simulation.simulation.PATH_DATA_MCAD_RESULTS",
        tmp_path,
    )
    calls: list[tuple[int, int]] = []

    def run_fake_simulation(self: simulations.MCADSimulation) -> None:
        first_csv: Path = tmp_path / "one.csv"
        second_csv: Path = tmp_path / "two.csv"

        assert self.csv_path.is_file()
        assert self.csv_path.with_suffix(".csv.meta.json").is_file()

        if self.csv_path == first_csv:
            assert not second_csv.exists()
            assert not second_csv.with_suffix(".csv.meta.json").exists()
        elif self.csv_path == second_csv:
            assert first_csv.is_file()
            assert first_csv.with_suffix(".csv.meta.json").is_file()
        else:
            raise AssertionError(f"Unexpected CSV path: {self.csv_path}")

        geometry_count = sum(1 for _ in self.geometry_generator)
        count = geometry_count * len(self.operating_map)
        calls.append((count, self.batch_size))
        self._report_progress("simulated", count)

    monkeypatch.setattr(simulations.MCADSimulation, "run", run_fake_simulation)
    pipeline = pipelines.SimulationPipeline(
        [(1, 1), (1, 2)],
        start_eduvpn=False,
        gen_type="v2",
        csv_paths=[tmp_path / "one.csv", tmp_path / "two.csv"],
        num_workers=1,
    )
    try:
        pipeline.service.prepare_output()

        for filename in ("one.csv", "two.csv"):
            csv_path: Path = tmp_path / filename
            assert not csv_path.exists()
            assert not csv_path.with_suffix(".csv.meta.json").exists()

        pipeline.service.run()
        states: list[tuple[int, int, int]] = []
        for _ in range(6):
            event = pipeline.progress_queue.get(timeout=5)
            value, _, _ = pipeline._queue_adapter(event)
            states.append((value, pipeline.overall_value, pipeline.max_progress_steps))
        points = math.prod(MAP_SIZE)
        assert calls == [(points, points), (2 * points, points)]
        assert states == [
            (0, 0, points),
            (points, 0, points),
            (points, 1, points),
            (0, 1, 2 * points),
            (2 * points, 1, 2 * points),
            (2 * points, 2, 2 * points),
        ]
    finally:
        pipeline._close_queues()


def test_failed_future_propagates_and_reports_no_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata: SimulationMetadata,
    event_queues: EventQueues,
) -> None:
    monkeypatch.setattr(
        "src.modules.simulation.simulation.build_simulation_metadata",
        lambda: metadata,
    )
    monkeypatch.setattr(
        "src.modules.simulation.simulation.PATH_DATA_MCAD_RESULTS",
        tmp_path,
    )

    progress, logs = event_queues

    instance = simulations.MCADSimulation(
        geometry_generator=(geometry for geometry in ()),
        operating_map=np.array([[0.5, 1.0, 0.0]]),
        batch_size=1,
        progress_queue=progress,
        log_queue=logs,
        csv_path=tmp_path / "out.csv",
        num_workers=1,
    )

    reference = ReferenceRunResult(1000.0, 2.0, 100.0)

    operating_point = simulations.OperatingPoint(
        parameters={"Pole_Number": 2.0},
        reference=reference,
        base_speed_rpm=reference.calculate_base_speed_rpm(800.0),
        dc_bus_voltage_v=800.0,
        shaft_speed_norm=0.5,
    )

    future: Future[tuple[list[float] | None, bool]] = Future()
    error = RuntimeError("simulation failure")
    future.set_exception(error)

    request = simulations.PendingRequest(
        schema=("Pole_Number",),
        point=np.array([2.0]),
        future=future,
        operating_point=operating_point,
    )

    with pytest.raises(RuntimeError) as captured:
        instance._collect_results([request])

    assert captured.value is error
    assert instance.dataframes == {}
