from __future__ import annotations

import logging
import random
import sys
import time
from dataclasses import dataclass
from importlib.metadata import version
from multiprocessing import Process
from multiprocessing.queues import Queue as ProcessQueue

import numpy as np
import optuna
import torch

from src.config.config_logging import setup_worker_logging
from src.config.config_mcad_parameters import (
    MACHINE_DEFINITION,
    MAP_SIZE,
    OUTPUT_NEURON_LIST,
)
from src.config.config_paths import PATH_DATA
from src.modules.data_managment.experiment_utils import (
    canonical_hash,
    file_sha256,
    validate_search_settings,
)
from src.modules.neuronal_networks.manipulating_data import (
    TrainValidationTestDataTensors,
)
from src.modules.neuronal_networks.nn_training_torch import NNTrainer
from src.modules.neuronal_networks.NN_types.mlp_with_torch import MLP
from src.modules.neuronal_networks.surrogate_model import SurrogateModelTrainer
from src.modules.neuronal_networks.testing import calc_mse, calc_r_squared_report
from src.modules.Pipeline_and_GUI.events import EventType, PipelineEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationResult:
    hidden_layers: tuple[int, ...]
    mse: float | None
    r_squared: float | None
    trial_number: int


def _optuna_worker(
    data: TrainValidationTestDataTensors,
    trainer: NNTrainer,
    max_num_layers: int,
    max_num_neurons_per_layer: int,
    neuron_step: int,
    n_trials: int,
    progress_queue: ProcessQueue[PipelineEvent],
    study_name: str,
    storage: str,
    seed: int,
    objective_seed: int,
    log_queue: ProcessQueue[logging.LogRecord] | None = None,
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

        logger.info("Starting optimization worker with seed %d", seed)

        study: optuna.Study = optuna.load_study(
            study_name=study_name,
            storage=storage,
            sampler=optuna.samplers.TPESampler(seed=seed),
        )

        objective: Objective = Objective(
            data=data,
            trainer=trainer,
            max_num_layers=max_num_layers,
            max_num_neurons_per_layer=max_num_neurons_per_layer,
            neuron_step=neuron_step,
            seed=objective_seed,
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

        logger.info("Optimization worker with seed %d completed", seed)

    except BaseException:
        logger.exception("Optimization worker with seed %d failed", seed)
        raise


class NNOptimizer:
    def __init__(
        self,
        data: TrainValidationTestDataTensors,
        progress_queue: ProcessQueue[PipelineEvent],
        trainer: NNTrainer,
        max_num_layers: int,
        max_num_neurons_per_layer: int,
        neuron_step: int,
        num_workers: int = 4,
        n_trials: int = 50,
        seed: int = 42,
        worker_timeout_seconds: float | None = None,
        log_queue: ProcessQueue[logging.LogRecord] | None = None,
    ) -> None:
        validate_search_settings(
            max_num_layers,
            max_num_neurons_per_layer,
            neuron_step,
            n_trials,
            num_workers,
        )

        if type(seed) is not int or not 0 <= seed <= 2**32 - num_workers:
            raise ValueError("Invalid seed")
        if worker_timeout_seconds is not None and (
            not np.isfinite(worker_timeout_seconds) or worker_timeout_seconds <= 0
        ):
            raise ValueError("Worker timeout must be positive")

        self.data: TrainValidationTestDataTensors = data
        self.progress_queue: ProcessQueue[PipelineEvent] = progress_queue
        self.log_queue: ProcessQueue[logging.LogRecord] | None = log_queue
        self.trainer = trainer
        self.max_num_layers = max_num_layers
        self.max_num_neurons_per_layer = max_num_neurons_per_layer
        self.step = neuron_step
        self.num_workers = num_workers
        self.n_trials = n_trials
        self.seed = seed
        self.worker_timeout_seconds = worker_timeout_seconds
        self.result: OptimizationResult | None = None
        self.experiment_config = self._experiment_config()
        self.fingerprint = canonical_hash(self.experiment_config)
        self.study_name = "layer_" + self.fingerprint
        self.experiment_dir = PATH_DATA / "optimization" / self.fingerprint
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        database = (self.experiment_dir / "study.sqlite3").resolve().as_posix()
        self.storage = "sqlite:///" + database

    def run(self) -> None:
        self._prepare_study()
        self._run_workers()

        study = optuna.load_study(study_name=self.study_name, storage=self.storage)
        best_trial = study.best_trial

        hidden_layers = tuple(
            best_trial.params[f"neurons_{i}"]
            for i in range(best_trial.params["n_layers"])
        )

        self.result = OptimizationResult(
            hidden_layers=hidden_layers,
            mse=float(best_trial.value),
            r_squared=best_trial.user_attrs.get("r2_mean"),
            trial_number=best_trial.number,
        )

    def _experiment_config(self) -> dict[str, object]:
        root = PATH_DATA.parent
        relevant_sources = (
            "src/config/config_mcad_parameters.py",
            "src/modules/neuronal_networks/manipulating_data.py",
            "src/modules/neuronal_networks/nn_training_torch.py",
            "src/modules/neuronal_networks/surrogate_model.py",
            "src/modules/neuronal_networks/testing.py",
            "src/modules/neuronal_networks/NN_types/mlp_with_torch.py",
            "src/modules/neuronal_networks/optimization/optimizer.py",
        )
        return {
            "version": 1,
            "sources": self.data.source_hashes,
            "split_ids": self.data.split_machine_ids,
            "map_axes": {k: v.tolist() for k, v in self.data.map_axes.items()},
            "target_minimum": self.data.target_minimum.tolist(),
            "target_denominator": self.data.target_denominator.tolist(),
            "feature_names": list(MACHINE_DEFINITION),
            "channel_names": list(OUTPUT_NEURON_LIST),
            "workers": self.num_workers,
            "model_output_dims": list(MAP_SIZE) + [1],
            "training": {
                "learning_rate": self.trainer.learning_rate,
                "weight_decay": self.trainer.weight_decay,
                "epochs": self.trainer.num_epochs,
                "patience": self.trainer.patience,
            },
            "search": {
                "max_layers": self.max_num_layers,
                "max_neurons": self.max_num_neurons_per_layer,
                "step": self.step,
            },
            "seed": self.seed,
            "python": list(sys.version_info[:3]),
            "package_versions": {
                name: version(name) for name in ("torch", "optuna", "numpy", "pandas")
            },
            "implementation": {
                name: file_sha256(root / name) for name in relevant_sources
            },
        }

    def _prepare_study(self) -> None:
        study: optuna.Study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction="minimize",
            load_if_exists=True,
        )

        existing_fingerprint = study.user_attrs.get("experiment_fingerprint")

        if existing_fingerprint is None and study.trials:
            raise ValueError(
                "The study in question contains trials without an experiment ID."
            )

        if (
            existing_fingerprint is not None
            and existing_fingerprint != self.fingerprint
        ):
            raise ValueError("The study forms part of a different experiment.")

        study.set_user_attr(
            "experiment_fingerprint",
            self.fingerprint,
        )
        study.set_user_attr(
            "experiment_config",
            self.experiment_config,
        )

    def _run_workers(self) -> None:
        processes: list[Process] = []
        deadline = (
            None
            if self.worker_timeout_seconds is None
            else time.monotonic() + self.worker_timeout_seconds
        )

        try:
            for index in range(self.num_workers):
                process = Process(
                    target=_optuna_worker,
                    args=(
                        self.data,
                        self.trainer,
                        self.max_num_layers,
                        self.max_num_neurons_per_layer,
                        self.step,
                        self.n_trials,
                        self.progress_queue,
                        self.study_name,
                        self.storage,
                        self.seed + index,
                        self.seed,
                        self.log_queue,
                    ),
                )

                processes.append(process)
                process.start()

            while True:
                for process in processes:
                    process.join(timeout=0.1)
                    if process.exitcode not in (None, 0):
                        raise RuntimeError(
                            f"Worker {process.pid} failed: {process.exitcode}"
                        )
                if all(p.exitcode is not None for p in processes):
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Optimization worker budget exceeded")

        finally:
            original_error = sys.exc_info()[1]
            try:
                self._cleanup_workers(processes)
            except Exception as cleanup_error:
                if original_error is None:
                    raise
                original_error.add_note(f"Cleanup error: {cleanup_error!r}")
                logger.exception("Worker cleanup also fail")

    @staticmethod
    def _cleanup_workers(processes: list[Process]) -> None:
        started_processes = [
            process for process in processes if process.pid is not None
        ]
        errors = []
        for process in started_processes:
            try:
                if process.is_alive():
                    process.terminate()
            except Exception as exc:
                errors.append(exc)

        for process in started_processes:
            try:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
                if process.is_alive():
                    raise RuntimeError(f"Worker {process.pid} still alive")
                process.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("Worker cleanup failed", errors)


class Objective:
    def __init__(
        self,
        data: TrainValidationTestDataTensors,
        trainer: NNTrainer,
        max_num_layers: int,
        max_num_neurons_per_layer: int,
        neuron_step: int,
        seed: int,
        num_workers: int = 4,
    ) -> None:
        self.data = data
        self.max_num_layers = max_num_layers
        self.max_num_neurons_per_layer = max_num_neurons_per_layer
        self.step = neuron_step
        self.num_workers = num_workers
        self.trainer = trainer
        self.seed = seed

    def __call__(self, trial: optuna.Trial) -> float:
        hidden_layer = self._get_layer(trial)
        torch.manual_seed((self.seed + trial.number) % 2**32)

        mlp = MLP(
            num_input_neurons=len(MACHINE_DEFINITION),
            output_dims=(*MAP_SIZE, 1),
            hidden_layer_neurons=hidden_layer,
        )
        trained_surrogate_model = SurrogateModelTrainer(self.trainer).fit(
            mlp=mlp,
            train_data=self.data.train,
            validation_data=self.data.validation,
        )

        map_pred_vali = trained_surrogate_model.get_prediction(
            geo_tensor=self.data.validation.geometry_tensor,
        )
        mse = calc_mse(self.data.validation.map_tensor, map_pred_vali)
        r2 = calc_r_squared_report(self.data.validation.map_tensor, map_pred_vali)
        trial.set_user_attr("r2_mean", r2.mean)
        trial.set_user_attr("r2_per_channel", list(r2.per_channel))
        return mse

    def _get_layer(
        self,
        trial: optuna.Trial,
    ) -> tuple[int, ...]:
        count = trial.suggest_int("n_layers", 1, self.max_num_layers)
        values = list(range(self.step, self.max_num_neurons_per_layer + 1, self.step))
        return tuple(
            trial.suggest_categorical(f"neurons_{i}", values) for i in range(count)
        )
