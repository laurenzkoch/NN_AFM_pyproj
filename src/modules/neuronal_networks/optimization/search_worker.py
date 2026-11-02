from __future__ import annotations

import logging
import random
from multiprocessing.queues import Queue as ProcessQueue
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna.storages import JournalStorage

from src.config.config_logging import setup_worker_logging
from src.config.config_mcad_parameters import OUTPUT_NEURON_LIST
from src.modules.neuronal_networks.data_types import (
    TrainValidationTestDataTensors,
)
from src.modules.neuronal_networks.nn_training_torch import NNTrainer
from src.modules.neuronal_networks.NN_types.cnn_autoencoder import (
    Conv3dAutoencoder,
)
from src.modules.neuronal_networks.optimization.journal_storage import (
    create_journal_storage,
)
from src.modules.neuronal_networks.optimization.objective import Objective
from src.modules.neuronal_networks.optimization.optimization_types import (
    NetworkSearchSpace,
    NetworkType,
)
from src.modules.runtime.events import EventType, PipelineEvent

logger = logging.getLogger(__name__)


def optuna_worker(
    data: TrainValidationTestDataTensors,
    trainer: NNTrainer,
    network_type: NetworkType,
    channel_index: int,
    search_space: NetworkSearchSpace,
    n_trials: int,
    progress_queue: ProcessQueue[PipelineEvent],
    study_name: str,
    journal_path: Path,
    seed: int,
    objective_seed: int,
    log_queue: ProcessQueue[logging.LogRecord] | None = None,
    trained_autoencoder: Conv3dAutoencoder | None = None,
) -> None:
    if log_queue is not None:
        setup_worker_logging(log_queue)
        optuna.logging.disable_default_handler()
        optuna.logging.enable_propagation()

    try:
        torch.set_num_threads(1)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        channel_name: str = OUTPUT_NEURON_LIST[channel_index]

        logger.info(
            "Starting %s optimization worker: channel=%s, seed=%d",
            network_type,
            channel_name,
            seed,
        )
        storage: JournalStorage = create_journal_storage(journal_path)
        study: optuna.Study = optuna.load_study(
            study_name=study_name,
            storage=storage,
            sampler=optuna.samplers.TPESampler(seed=seed),
        )

        objective: Objective = Objective(
            data=data,
            trainer=trainer,
            network_type=network_type,
            channel_index=channel_index,
            search_space=search_space,
            seed=objective_seed,
            trained_autoencoder=trained_autoencoder,
        )

        def _progress_callback(
            _study: optuna.Study,
            _trial: optuna.trial.FrozenTrial,
        ) -> None:
            progress_queue.put(
                PipelineEvent(
                    event_type=EventType.PROGRESS,
                    increment=1,
                    outcome="trial_completed",
                )
            )

        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[_progress_callback],
        )

        logger.info(
            "Completed %s optimization worker: channel=%s, seed=%d",
            network_type,
            channel_name,
            seed,
        )

    except BaseException:
        logger.exception(
            "Optimization worker failed: type=%s, channel_index=%d, seed=%d",
            network_type,
            channel_index,
            seed,
        )
        raise
