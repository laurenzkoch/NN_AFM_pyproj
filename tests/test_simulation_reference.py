import logging
from collections.abc import Callable
from concurrent.futures import Future
from multiprocessing.queues import Queue as ProcessQueue
from pathlib import Path
from typing import Protocol, TypedDict
from unittest.mock import Mock

import numpy as np
import pytest

from src.config.config_mcad_parameters import (
    CSV_PARAMETERS,
    MCAD_RESULT_PARAMETERS,
)
from src.modules.data_managment.csv_io import import_data_from_csv
from src.modules.data_managment.simulation_metadata import SimulationMetadata
from src.modules.domain.reference_run import ReferenceRunResult
from src.modules.runtime.events import PipelineEvent
from src.modules.simulation import simulation as simulations
from src.modules.simulation import workers
from tests.conftest import make_frame


class PoolSettings(TypedDict):
    failed_reference: bool
    error_stage: str | None
    rejected_bore: float | None


type EventQueues = tuple[
    ProcessQueue[PipelineEvent],
    ProcessQueue[logging.LogRecord],
]

type PoolDouble = tuple[
    list[tuple[str, int]],
    list[tuple[int, str, dict[str, float]]],
    PoolSettings,
]


class SimulationFactory(Protocol):
    def __call__(
        self,
        geometry_count: int = 1,
    ) -> simulations.MCADSimulation: ...


@pytest.fixture
def simulation_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata: SimulationMetadata,
    event_queues: EventQueues,
) -> SimulationFactory:

    monkeypatch.setattr(
        "src.modules.simulation.simulation.build_simulation_metadata",
        lambda: metadata,
    )
    monkeypatch.setattr(
        "src.modules.simulation.simulation.PATH_DATA_MCAD_RESULTS",
        tmp_path,
    )

    progress, logs = event_queues

    def create(
        geometry_count: int = 1,
    ) -> simulations.MCADSimulation:
        geometries = ({"Stator_Bore": 70.0 + index} for index in range(geometry_count))

        return simulations.MCADSimulation(
            geometry_generator=geometries,
            operating_map=np.array(
                [
                    [0.5, 3.0, -4.0],
                    [1.0, 5.0, 0.0],
                ]
            ),
            batch_size=4,
            progress_queue=progress,
            log_queue=logs,
            csv_path=tmp_path / "out.csv",
            num_workers=2,
        )

    return create


@pytest.fixture
def pool_double(
    monkeypatch: pytest.MonkeyPatch,
) -> PoolDouble:

    lifecycle: list[tuple[str, int]] = []
    submissions: list[tuple[int, str, dict[str, float]]] = []

    settings: PoolSettings = {
        "failed_reference": False,
        "error_stage": None,
        "rejected_bore": None,
    }

    template = make_frame(1).iloc[0].to_dict()
    pool_count = 0

    def create_pool(**kwargs: object) -> Mock:
        nonlocal pool_count

        assert kwargs["max_workers"] == 2
        assert kwargs["initializer"] is (
            simulations.mcad_instance_handling.init_mcad_instance
        )

        pool_id = pool_count
        pool_count += 1

        lifecycle.append(("opened", pool_id))
        pool = Mock()

        def submit(
            function: Callable[..., object],
            parameters: dict[str, float],
            *,
            log_context: str = "",
        ) -> Future[object]:

            assert "batch=" in log_context
            assert "geometry=" in log_context
            assert "task=" in log_context
            assert function in (
                workers.execute_simulation,
                workers.execute_reference_simulation,
            )

            true_for_reference_run = function is workers.execute_reference_simulation

            stage = "reference" if true_for_reference_run else "load"

            submissions.append((pool_id, stage, dict(parameters)))

            future: Future[object] = Future()

            # Unexpected worker errors must still abort the simulation.
            if settings["error_stage"] == stage:
                future.set_exception(RuntimeError(f"{stage} failure"))

            elif true_for_reference_run:
                rejected = (
                    settings["failed_reference"]
                    or parameters["Stator_Bore"] == settings["rejected_bore"]
                )

                if rejected:
                    future.set_result(None)
                else:
                    ke = 2.0 + parameters["Stator_Bore"] - 70.0

                    future.set_result(
                        ReferenceRunResult(
                            reference_speed_rpm=1000.0,
                            back_emf_constant_ll_peak_vs_per_rad=ke,
                            peak_back_emf_phase_v=100.0,
                        )
                    )

            else:
                values = {
                    **template,
                    **parameters,
                }

                row = [values[name] for name in MCAD_RESULT_PARAMETERS]
                future.set_result((row, True))

            return future

        def shutdown(wait: bool = True) -> None:
            assert wait is True
            lifecycle.append(("closed", pool_id))

        def kill_workers() -> None:
            lifecycle.append(("killed", pool_id))

        pool.submit.side_effect = submit
        pool.shutdown.side_effect = shutdown
        pool.kill_workers.side_effect = kill_workers

        return pool

    monkeypatch.setattr(
        "src.modules.simulation.simulation.ProcessPoolExecutor",
        create_pool,
    )

    return lifecycle, submissions, settings


def test_pool_restarts_after_each_geometry_batch(
    simulation_factory: SimulationFactory,
    pool_double: PoolDouble,
) -> None:

    lifecycle, submissions, _settings = pool_double
    instance = simulation_factory(geometry_count=5)

    instance.prepare_output()
    instance.run()

    # Two geometries per pool; the final pool handles one.
    assert lifecycle == [
        ("opened", 0),
        ("closed", 0),
        ("opened", 1),
        ("closed", 1),
        ("opened", 2),
        ("closed", 2),
    ]

    reference_pool_ids = [
        pool_id for pool_id, stage, _parameters in submissions if stage == "reference"
    ]

    load_pool_ids = [
        pool_id for pool_id, stage, _parameters in submissions if stage == "load"
    ]

    assert reference_pool_ids == [0, 0, 1, 1, 2]
    assert load_pool_ids == [0] * 4 + [1] * 4 + [2] * 2

    assert [stage for _, stage, _ in submissions] == (["reference", "load", "load"] * 5)

    frame = import_data_from_csv(instance.csv_path)

    assert frame.columns.tolist() == CSV_PARAMETERS
    assert len(frame) == 10

    assert frame["ReferenceKeFundamental"].tolist() == [
        2.0,
        2.0,
        3.0,
        3.0,
        4.0,
        4.0,
        5.0,
        5.0,
        6.0,
        6.0,
    ]

    np.testing.assert_allclose(
        frame["ShaftSpeed"] / frame["BaseSpeedRPM"],
        frame["ShaftSpeedNorm"],
        rtol=0.0,
        atol=1e-10,
    )


def test_failed_reference_skips_its_operating_map(
    simulation_factory: SimulationFactory,
    pool_double: PoolDouble,
    event_queues: EventQueues,
) -> None:

    lifecycle, submissions, settings = pool_double
    settings["failed_reference"] = True

    instance = simulation_factory()
    instance.prepare_output()
    instance.run()

    assert lifecycle == [
        ("opened", 0),
        ("closed", 0),
    ]

    assert len(submissions) == 1
    assert submissions[0][1] == "reference"

    assert import_data_from_csv(instance.csv_path).empty

    progress, _logs = event_queues
    event = progress.get(timeout=5)

    assert event.outcome == "dropped"
    assert event.increment == 2


@pytest.mark.parametrize("stage", ["reference", "load"])
def test_worker_error_kills_pool_and_stops_further_work(
    stage: str,
    simulation_factory: SimulationFactory,
    pool_double: PoolDouble,
) -> None:

    lifecycle, submissions, settings = pool_double
    settings["error_stage"] = stage

    instance = simulation_factory(geometry_count=5)
    instance.prepare_output()

    with pytest.raises(
        RuntimeError,
        match=f"{stage} failure",
    ):
        instance.run()

    assert lifecycle == [
        ("opened", 0),
        ("killed", 0),
    ]

    # No subsequent geometry or replacement pool may start.
    assert (
        sum(submission_stage == "reference" for _, submission_stage, _ in submissions)
        == 1
    )

    assert import_data_from_csv(instance.csv_path).empty


@pytest.mark.parametrize("point_count", [2, 125])
def test_rejected_reference_drops_map_and_continues(
    point_count: int,
    simulation_factory: SimulationFactory,
    pool_double: PoolDouble,
    event_queues: EventQueues,
    caplog: pytest.LogCaptureFixture,
) -> None:
    lifecycle, submissions, settings = pool_double
    settings["rejected_bore"] = 70.0

    instance = simulation_factory(geometry_count=2)
    instance.operating_map = np.column_stack(
        (
            np.linspace(0.5, 1.0, point_count),
            np.full(point_count, 3.0),
            np.full(point_count, -4.0),
        )
    )
    instance.batch_size = 2 * point_count

    instance.prepare_output()
    instance.run()

    # Rejection must not kill the pool or stop the next geometry.
    assert lifecycle == [
        ("opened", 0),
        ("closed", 0),
    ]

    reference_designs = [
        parameters["Stator_Bore"]
        for _, stage, parameters in submissions
        if stage == "reference"
    ]
    assert reference_designs == [70.0, 71.0]

    load_designs = [
        parameters["Stator_Bore"]
        for _, stage, parameters in submissions
        if stage == "load"
    ]
    assert load_designs == [71.0] * point_count

    frame = import_data_from_csv(instance.csv_path)
    assert len(frame) == point_count
    assert (frame["Stator_Bore"] == 71.0).all()

    progress, _logs = event_queues
    events = [progress.get(timeout=5) for _ in range(point_count + 1)]

    dropped = [event for event in events if event.outcome == "dropped"]
    assert len(dropped) == 1
    assert dropped[0].increment == point_count

    simulated_count = sum(
        event.increment for event in events if event.outcome == "simulated"
    )
    assert simulated_count == point_count

    rejection_logs = [
        record.getMessage()
        for record in caplog.records
        if "Reference rejected; dropping operating map" in record.getMessage()
    ]
    assert len(rejection_logs) == 1
    assert f"dropped_points={point_count}" in rejection_logs[0]
    assert "'Stator_Bore': 70.0" in rejection_logs[0]
    assert "geometry=" in rejection_logs[0]
