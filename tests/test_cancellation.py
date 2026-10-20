import logging
from collections.abc import Generator, Iterable
from concurrent.futures import FIRST_COMPLETED, Future
from multiprocessing.queues import Queue as ProcessQueue
from pathlib import Path

import pytest

from src.modules.data_managment.simulation_metadata import SimulationMetadata
from src.modules.neuronal_networks.manipulating_data import (
    TrainValidationTestDataTensors,
)
from src.modules.neuronal_networks.nn_training_torch import NNTrainer
from src.modules.neuronal_networks.optimization.optimizer import NNOptimizer
from src.modules.Pipeline_and_GUI.cancellation import (
    CancellationToken,
    PipelineCancelledError,
)
from src.modules.Pipeline_and_GUI.events import PipelineEvent
from src.modules.simulation import simulation as simulations

type SimulationFuture = Future[tuple[list[float] | None, bool]]


def test_cancellation_token_rejects_further_work() -> None:
    token: CancellationToken = CancellationToken()

    assert not token.is_requested()
    token.raise_if_requested()

    token.request()
    token.request()

    assert token.is_requested()

    with pytest.raises(PipelineCancelledError):
        token.raise_if_requested()


def test_simulation_cancellation_kills_pool_and_stops_batches(
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

    token: CancellationToken = CancellationToken()
    calls: list[str] = []
    submitted: list[SimulationFuture] = []

    class FakePool:
        def __init__(self, **_kwargs: object) -> None:
            calls.append("created")

        @staticmethod
        def submit(
            _function: object,
            _parameters: dict[str, float],
        ) -> SimulationFuture:
            future: SimulationFuture = Future()
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
        futures: Iterable[SimulationFuture],
        timeout: float | None = None,
        return_when: str = FIRST_COMPLETED,
    ) -> tuple[set[SimulationFuture], set[SimulationFuture]]:
        assert timeout == 0.1
        assert return_when == FIRST_COMPLETED
        calls.append("waiting")
        token.request()
        return set(), set(futures)

    monkeypatch.setattr(
        "src.modules.simulation.simulation.ProcessPoolExecutor",
        FakePool,
    )
    monkeypatch.setattr(
        "src.modules.simulation.simulation.wait",
        wait_then_cancel,
    )

    progress, logs = event_queues

    points: Generator[dict[str, float]] = (
        {"Pole_Number": value} for value in (2.0, 4.0)
    )

    simulation: simulations.MCADSimulation = simulations.MCADSimulation(
        point_generator=points,
        batch_size=1,
        progress_queue=progress,
        log_queue=logs,
        csv_path=tmp_path / "cancelled.csv",
        num_workers=1,
        cancellation=token,
    )
    simulation.prepare_output()

    with pytest.raises(PipelineCancelledError):
        simulation.run()

    assert calls == ["created", "waiting", "killed"]
    assert len(submitted) == 1
    assert not submitted[0].done()
    assert simulation.dataframes == {}


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
        max_num_layers=1,
        max_num_neurons_per_layer=4,
        neuron_step=4,
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
    assert optimizer.result is None
