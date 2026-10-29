import logging
from collections.abc import Generator, Iterable
from concurrent.futures import FIRST_COMPLETED, Future
from multiprocessing.queues import Queue as ProcessQueue
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from src.modules.data_managment.simulation_metadata import SimulationMetadata
from src.modules.neuronal_networks.manipulating_data import (
    TrainValidationTestDataTensors,
)
from src.modules.neuronal_networks.nn_training_torch import NNTrainer
from src.modules.neuronal_networks.optimization.optimization_types import (
    NetworkSearchSpace,
)
from src.modules.neuronal_networks.optimization.optimizer import NNOptimizer
from src.modules.Pipeline_and_GUI.cancellation import (
    CancellationToken,
    PipelineCancelledError,
)
from src.modules.Pipeline_and_GUI.events import PipelineEvent
from src.modules.simulation import simulation as simulations
from src.modules.simulation.reference_run import ReferenceRunResult

type WorkerResult = tuple[list[float] | None, bool] | ReferenceRunResult | None

type WorkerFuture = Future[WorkerResult]


def test_cancellation_token_rejects_further_work() -> None:
    token: CancellationToken = CancellationToken()

    assert not token.is_requested()
    token.raise_if_requested()

    token.request()
    token.request()

    assert token.is_requested()

    with pytest.raises(PipelineCancelledError):
        token.raise_if_requested()


@pytest.mark.parametrize("cancel_stage", ["reference", "load"])
def test_simulation_cancellation_kills_pool_and_stops_batches(
    cancel_stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata: SimulationMetadata,
    event_queues: tuple[
        ProcessQueue[PipelineEvent],
        ProcessQueue[logging.LogRecord],
    ],
) -> None:
    monkeypatch.setattr(
        "src.modules.simulation.simulation.build_simulation_metadata",
        lambda: metadata,
    )
    monkeypatch.setattr(
        "src.modules.simulation.simulation.PATH_DATA_MCAD_RESULTS",
        tmp_path,
    )

    token = CancellationToken()

    calls: list[str] = []
    submitted: list[WorkerFuture] = []
    submitted_stages: list[str] = []

    class FakePool:
        def __init__(self, **_kwargs: object) -> None:
            calls.append("created")

        @staticmethod
        def submit(
            function: object,
            _parameters: dict[str, float],
            *,
            log_context: str = "",
        ) -> WorkerFuture:

            assert "batch=" in log_context
            assert "geometry=" in log_context
            assert "task=" in log_context

            assert function in (
                simulations._execute_simulation,
                simulations._execute_reference_simulation,
            )

            true_for_reference_run = (
                function is simulations._execute_reference_simulation
            )

            future: WorkerFuture = Future()

            stage = "reference" if true_for_reference_run else "load"
            submitted_stages.append(stage)

            if true_for_reference_run and cancel_stage == "load":
                # Let the reference finish so the test reaches
                # the wait for the load calculation.
                future.set_result(
                    ReferenceRunResult(
                        reference_speed_rpm=1000.0,
                        back_emf_constant_ll_peak_vs_per_rad=2.0,
                        peak_back_emf_phase_v=100.0,
                    )
                )
            else:
                future.set_running_or_notify_cancel()

            submitted.append(future)
            return future

        @staticmethod
        def kill_workers() -> None:
            calls.append("killed")

        @staticmethod
        def shutdown(wait: bool = True) -> None:
            calls.append("shutdown")

    def wait_then_cancel(
        futures: Iterable[WorkerFuture],
        timeout: float | None = None,
        return_when: str = FIRST_COMPLETED,
    ) -> tuple[set[WorkerFuture], set[WorkerFuture]]:
        assert timeout == 0.1
        assert return_when == FIRST_COMPLETED

        future_set = set(futures)

        completed = {future for future in future_set if future.done()}

        if completed:
            return completed, future_set - completed

        calls.append("waiting")
        token.request()

        return set(), future_set

    monkeypatch.setattr(
        "src.modules.simulation.simulation.ProcessPoolExecutor",
        FakePool,
    )
    monkeypatch.setattr(
        "src.modules.simulation.simulation.wait",
        wait_then_cancel,
    )

    progress, logs = event_queues

    geometries: Generator[dict[str, float]] = (
        {"Stator_Bore": value} for value in (70.0, 71.0)
    )

    instance = simulations.MCADSimulation(
        geometry_generator=geometries,
        operating_map=np.array([[0.5, 3.0, -4.0]]),
        batch_size=1,
        progress_queue=progress,
        log_queue=logs,
        csv_path=tmp_path / "cancelled.csv",
        num_workers=1,
        cancellation=token,
    )
    instance.prepare_output()

    with pytest.raises(PipelineCancelledError):
        instance.run()

    assert calls == [
        "created",
        "waiting",
        "killed",
    ]

    expected_stages = (
        ["reference"] if cancel_stage == "reference" else ["reference", "load"]
    )

    assert submitted_stages == expected_stages

    unfinished = [future for future in submitted if not future.done()]

    assert len(unfinished) == 1
    assert instance.dataframes == {}


def test_optimizer_cancellation_cleans_up_all_started_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tensor_data: TrainValidationTestDataTensors,
    event_queues: tuple[
        ProcessQueue[PipelineEvent],
        ProcessQueue[logging.LogRecord],
    ],
) -> None:
    token: CancellationToken = CancellationToken()

    class FakeProcess:
        def __init__(self, **_kwargs: object) -> None:
            self.pid: int | None = None
            self.exitcode: int | None = None
            self.running: bool = False
            self.closed: bool = False
            created.append(self)

        def start(self) -> None:
            self.pid = 1000 + len(created)
            self.running = True

        def join(self, timeout: float | None = None) -> None:
            assert timeout is not None and timeout > 0
            if self.running:
                token.request()

        def is_alive(self) -> bool:
            return self.running

        def terminate(self) -> None:
            self.running = False
            self.exitcode = -15

        def kill(self) -> None:
            self.running = False
            self.exitcode = -9

        def close(self) -> None:
            assert not self.running
            self.closed = True

    created: list[FakeProcess] = []

    monkeypatch.setattr(
        "src.modules.neuronal_networks.optimization.optimizer.Process",
        FakeProcess,
    )
    monkeypatch.setattr(
        "src.modules.neuronal_networks.optimization.optimizer.PATH_DATA",
        tmp_path,
    )
    monkeypatch.setattr(
        "src.modules.neuronal_networks.optimization.optimizer."
        "NNOptimizer._experiment_config",
        lambda _self: {"test": True},
    )

    progress, logs = event_queues

    optimizer: NNOptimizer = NNOptimizer(
        data=tensor_data,
        progress_queue=progress,
        log_queue=logs,
        trainer=NNTrainer(num_epochs=1, patience=1),
        network_type="MLP",
        channel_index=0,
        search_space=NetworkSearchSpace(
            max_num_layers=1,
            max_num_units_per_layer=4,
            unit_step=4,
        ),
        num_workers=2,
        n_trials=1,
        cancellation=token,
    )

    with pytest.raises(PipelineCancelledError):
        optimizer._run_workers()

    assert len(created) == 2
    assert all(process.pid is not None for process in created)
    assert all(not process.is_alive() for process in created)
    assert all(process.closed for process in created)
    assert optimizer.channel_result is None


def test_requested_cancellation_prevents_pool_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata: SimulationMetadata,
    event_queues: tuple[
        ProcessQueue[PipelineEvent],
        ProcessQueue[logging.LogRecord],
    ],
) -> None:
    monkeypatch.setattr(
        "src.modules.simulation.simulation.build_simulation_metadata",
        lambda: metadata,
    )
    monkeypatch.setattr(
        "src.modules.simulation.simulation.PATH_DATA_MCAD_RESULTS",
        tmp_path,
    )

    pool_factory = Mock()

    monkeypatch.setattr(
        "src.modules.simulation.simulation.ProcessPoolExecutor",
        pool_factory,
    )

    token = CancellationToken()
    token.request()

    progress, logs = event_queues

    instance = simulations.MCADSimulation(
        geometry_generator=({"Stator_Bore": value} for value in (70.0, 71.0)),
        operating_map=np.array([[0.5, 3.0, -4.0]]),
        batch_size=1,
        progress_queue=progress,
        log_queue=logs,
        csv_path=tmp_path / "not_started.csv",
        num_workers=1,
        cancellation=token,
    )

    with pytest.raises(PipelineCancelledError):
        instance.run()

    pool_factory.assert_not_called()
