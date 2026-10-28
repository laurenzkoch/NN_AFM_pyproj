from __future__ import annotations

import logging
import math
from multiprocessing import get_context
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue as ProcessQueue
from pathlib import Path

import optuna
import pytest
from optuna.storages import JournalStorage

from src.modules.neuronal_networks.manipulating_data import (
    TrainValidationTestDataTensors,
)
from src.modules.neuronal_networks.nn_training_torch import NNTrainer
from src.modules.neuronal_networks.optimization.journal_storage import (
    create_journal_storage,
)
from src.modules.neuronal_networks.optimization.optimization_types import (
    NetworkSearchSpace,
)
from src.modules.neuronal_networks.optimization.optimizer import _optuna_worker
from src.modules.Pipeline_and_GUI.events import EventType, PipelineEvent


@pytest.mark.training
def test_worker_runs_requested_trials_once(
    tmp_path: Path,
    tensor_data: TrainValidationTestDataTensors,
    event_queues: tuple[ProcessQueue[PipelineEvent], ProcessQueue[logging.LogRecord]],
) -> None:
    progress, _ = event_queues
    journal_path: Path = tmp_path / "study.journal"
    storage: JournalStorage = create_journal_storage(journal_path)
    study = optuna.create_study(
        study_name="synthetic-worker", storage=storage, direction="minimize"
    )
    _optuna_worker(
        data=tensor_data,
        trainer=NNTrainer(num_epochs=2, patience=1),
        network_type="MLP",
        channel_index=0,
        search_space=NetworkSearchSpace(
            max_num_layers=1,
            max_num_units_per_layer=4,
            unit_step=4,
        ),
        n_trials=2,
        progress_queue=progress,
        study_name=study.study_name,
        journal_path=journal_path,
        seed=42,
        objective_seed=42,
    )
    completed: optuna.Study = optuna.load_study(
        study_name=study.study_name,
        storage=create_journal_storage(journal_path),
    )
    assert len(completed.trials) == 2
    assert all(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in completed.trials
    )
    for trial in completed.trials:
        objective_value: float | None = trial.value

        assert objective_value is not None
        assert math.isfinite(objective_value)
        assert objective_value >= 0.0

        assert trial.user_attrs["channel_index"] == 0
        assert trial.user_attrs["loss_scope"] == "full_map"
        assert trial.user_attrs["architecture"] == {
            "network_type": "MLP",
            "layers": [4],
            "latent_dim": None,
            "input_channels": 1,
        }
    events = [progress.get(timeout=5) for _ in range(2)]
    assert all(
        event.event_type == EventType.PROGRESS and event.increment == 1
        for event in events
    )
    assert all(event.outcome == "trial_completed" for event in events)


@pytest.mark.training
def test_workers_share_journal(
    tmp_path: Path,
    tensor_data: TrainValidationTestDataTensors,
    event_queues: tuple[
        ProcessQueue[PipelineEvent],
        ProcessQueue[logging.LogRecord],
    ],
) -> None:
    progress, _ = event_queues
    journal_path: Path = tmp_path / "parallel.journal"
    study_name: str = "parallel-worker-test"

    optuna.create_study(
        study_name=study_name,
        storage=create_journal_storage(journal_path),
        direction="minimize",
    )

    context = get_context("spawn")
    processes: list[BaseProcess] = []

    try:
        for worker_index in range(2):
            process: BaseProcess = context.Process(
                target=_optuna_worker,
                kwargs={
                    "data": tensor_data,
                    "trainer": NNTrainer(num_epochs=2, patience=1),
                    "network_type": "MLP",
                    "channel_index": 0,
                    "search_space": NetworkSearchSpace(
                        max_num_layers=1,
                        max_num_units_per_layer=4,
                        unit_step=4,
                    ),
                    "n_trials": 2,
                    "progress_queue": progress,
                    "study_name": study_name,
                    "journal_path": journal_path,
                    "seed": 42 + worker_index,
                    "objective_seed": 42,
                },
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join(timeout=60)
            assert not process.is_alive(), "Worker exceeded test timeout."
            assert process.exitcode == 0

        completed: optuna.Study = optuna.load_study(
            study_name=study_name,
            storage=create_journal_storage(journal_path),
        )

        assert len(completed.trials) == 4
        assert len({trial.number for trial in completed.trials}) == 4

        for trial in completed.trials:
            assert trial.state == optuna.trial.TrialState.COMPLETE
            assert trial.user_attrs["channel_index"] == 0

            objective_value: float | None = trial.value
            assert objective_value is not None
            assert math.isfinite(objective_value)

        assert journal_path.is_file()
        assert not list(tmp_path.glob("*.sqlite3"))

    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

            if process.is_alive():
                process.kill()
                process.join(timeout=5)

            if not process.is_alive():
                process.close()
