from __future__ import annotations

import copy
import hashlib
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from multiprocessing import Process
from multiprocessing.queues import Queue as ProcessQueue

import numpy as np
import optuna
import torch

from src.config.config_logging import setup_worker_logging
from src.config.config_mcad_parameters import (
    MACHINE_DEFINITION,
    OUTPUT_NEURON_LIST,
)
from src.config.config_paths import PATH_DATA
from src.modules.data_managment.experiment_utils import (
    canonical_hash,
    file_sha256,
    validate_search_settings,
)
from src.modules.neuronal_networks.manipulating_data import (
    MachineDataTensor,
    TrainValidationTestDataTensors,
)
from src.modules.neuronal_networks.nn_training_torch import NNTrainer
from src.modules.neuronal_networks.NN_types.cnn_autoencoder import (
    Conv3dAutoencoder,
)
from src.modules.neuronal_networks.NN_types.mlp_with_torch import MLP
from src.modules.neuronal_networks.optimization.optimization_types import (
    ChannelOptimizationResult,
    NetworkArchitecture,
    NetworkSearchSpace,
    NetworkType,
    checked_network_type,
)
from src.modules.Pipeline_and_GUI.cancellation import CancellationToken
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
    network_type: NetworkType,
    channel_index: int,
    search_space: NetworkSearchSpace,
    n_trials: int,
    progress_queue: ProcessQueue[PipelineEvent],
    study_name: str,
    storage: str,
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


class NNOptimizer:
    def __init__(
        self,
        data: TrainValidationTestDataTensors,
        progress_queue: ProcessQueue[PipelineEvent],
        trainer: NNTrainer,
        network_type: NetworkType,
        channel_index: int,
        search_space: NetworkSearchSpace,
        num_workers: int = 4,
        n_trials: int = 50,
        seed: int = 42,
        worker_timeout_seconds: float | None = None,
        log_queue: ProcessQueue[logging.LogRecord] | None = None,
        cancellation: CancellationToken | None = None,
        trained_autoencoder: Conv3dAutoencoder | None = None,
    ) -> None:
        validate_search_settings(
            search_space.max_num_layers,
            search_space.max_num_units_per_layer,
            search_space.unit_step,
            n_trials,
            num_workers,
        )

        if network_type not in ("MLP", "CNN"):
            raise ValueError("network_type must be 'MLP' or 'CNN'.")

        if type(channel_index) is not int or not 0 <= channel_index < len(
            OUTPUT_NEURON_LIST
        ):
            raise ValueError("Invalid output channel index.")

        if network_type == "CNN" and trained_autoencoder is not None:
            raise ValueError("CNN optimization cannot receive a trained autoencoder.")

        if type(seed) is not int or not 0 <= seed <= 2**32 - num_workers:
            raise ValueError("Invalid seed.")

        if worker_timeout_seconds is not None and (
            not np.isfinite(worker_timeout_seconds) or worker_timeout_seconds <= 0
        ):
            raise ValueError("Worker timeout must be finite and positive.")

        self.data: TrainValidationTestDataTensors = data
        self.progress_queue: ProcessQueue[PipelineEvent] = progress_queue
        self.log_queue: ProcessQueue[logging.LogRecord] | None = log_queue
        self.trainer: NNTrainer = trainer

        self.network_type: NetworkType = network_type
        self.channel_index: int = channel_index
        self.search_space: NetworkSearchSpace = search_space

        self.trained_autoencoder: Conv3dAutoencoder | None = copy.deepcopy(
            trained_autoencoder
        )

        if self.trained_autoencoder is not None:
            if self.trained_autoencoder.input_channels != 1:
                raise ValueError("The supplied CNN must process one output channel.")

            expected_shape: tuple[int, ...] = tuple(data.train.map_tensor.shape[1:-1])

            if tuple(self.trained_autoencoder.map_shape) != expected_shape:
                raise ValueError("The supplied CNN has an incompatible map shape.")

            self.trained_autoencoder.eval()
            self.trained_autoencoder.requires_grad_(False)

        self.cancellation: CancellationToken = (
            cancellation if cancellation is not None else CancellationToken()
        )

        self.num_workers: int = num_workers
        self.n_trials: int = n_trials
        self.seed: int = seed
        self.worker_timeout_seconds: float | None = worker_timeout_seconds

        self.channel_result: ChannelOptimizationResult | None = None

        self.experiment_config = self._experiment_config()
        self.fingerprint = canonical_hash(self.experiment_config)
        self.study_name = self.fingerprint

        self.experiment_dir = PATH_DATA / "optimization" / self.fingerprint
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        database: str = (self.experiment_dir / "study.sqlite3").resolve().as_posix()
        self.storage: str = "sqlite:///" + database

    def run(self) -> None:
        self.cancellation.raise_if_requested()
        self._prepare_study()

        self.cancellation.raise_if_requested()
        self._run_workers()

        self.cancellation.raise_if_requested()

        study: optuna.Study = optuna.load_study(
            study_name=self.study_name,
            storage=self.storage,
        )

        completed_trials = study.get_trials(
            deepcopy=False,
            states=(optuna.trial.TrialState.COMPLETE,),
        )

        if not completed_trials:
            raise RuntimeError(
                "Optimization finished without a completed trial "
                f"for channel {OUTPUT_NEURON_LIST[self.channel_index]}."
            )

        best_trial: optuna.trial.FrozenTrial = study.best_trial
        objective_value: float | None = best_trial.value

        if objective_value is None:
            raise ValueError("The best trial has no objective value.")

        best_mse: float = float(objective_value)

        if not np.isfinite(best_mse):
            raise ValueError("The best trial has a non-finite objective value.")

        raw_architecture: object = best_trial.user_attrs.get("architecture")

        if not isinstance(raw_architecture, dict):
            raise ValueError("The best trial has no valid architecture information.")

        if raw_architecture.get("network_type") != self.network_type:
            raise ValueError("The best trial belongs to a different network type.")

        if best_trial.user_attrs.get("channel_index") != self.channel_index:
            raise ValueError("The best trial belongs to a different output channel.")

        raw_layers: object = raw_architecture.get("layers")

        if (
            not isinstance(raw_layers, list)
            or not raw_layers
            or any(type(value) is not int for value in raw_layers)
        ):
            raise ValueError("Invalid layer sizes in the best trial.")

        raw_latent_dim: object = raw_architecture.get("latent_dim")

        if raw_latent_dim is not None and type(raw_latent_dim) is not int:
            raise ValueError("Invalid latent dimension in the best trial.")

        raw_input_channels: object = raw_architecture.get("input_channels")

        if type(raw_input_channels) is not int:
            raise ValueError("Invalid input channel count in the best trial.")

        architecture: NetworkArchitecture = NetworkArchitecture(
            network_type=checked_network_type(self.network_type),
            layers=tuple(raw_layers),
            latent_dim=raw_latent_dim,
            input_channels=raw_input_channels,
        )

        self.channel_result = ChannelOptimizationResult(
            channel_index=self.channel_index,
            architecture=architecture,
            mse=best_mse,
            r_squared=None,
            trial_number=best_trial.number,
        )

        logger.info(
            "Optimization completed: channel=%s, type=%s, "
            "layers=%s, latent_dim=%s, validation_mse=%g, scope=%s",
            OUTPUT_NEURON_LIST[self.channel_index],
            self.network_type,
            architecture.layers,
            architecture.latent_dim,
            best_mse,
            best_trial.user_attrs.get("loss_scope"),
        )

    def _autoencoder_config(self) -> dict[str, object] | None:
        autoencoder: Conv3dAutoencoder | None = self.trained_autoencoder

        if autoencoder is None:
            return None

        digest = hashlib.sha256()
        state: dict[str, torch.Tensor] = dict(autoencoder.state_dict())

        for name, value in sorted(state.items()):
            tensor: torch.Tensor = value.detach().cpu().contiguous()

            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"The supplied CNN contains non-finite values: {name}")

            descriptor: dict[str, object] = {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }

            digest.update(canonical_hash(descriptor).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())

        return {
            "input_channels": autoencoder.input_channels,
            "features": list(autoencoder.features),
            "latent_dim": autoencoder.latent_dim,
            "map_shape": list(autoencoder.map_shape),
            "kernel_size": autoencoder.kernel_size,
            "structure": repr(autoencoder),
            "state_sha256": digest.hexdigest(),
        }

    def _experiment_config(self) -> dict[str, object]:
        root = PATH_DATA.parent

        relevant_sources: tuple[str, ...] = (
            "src/config/config_mcad_parameters.py",
            "src/modules/neuronal_networks/manipulating_data.py",
            "src/modules/neuronal_networks/nn_training_torch.py",
            "src/modules/neuronal_networks/NN_types/mlp_with_torch.py",
            "src/modules/neuronal_networks/NN_types/cnn_autoencoder.py",
            "src/modules/neuronal_networks/optimization/optimization_types.py",
            "src/modules/neuronal_networks/optimization/optimizer.py",
        )

        map_shape: tuple[int, ...] = tuple(self.data.train.map_tensor.shape[1:-1])

        mlp_output_dims: tuple[int, ...] | None = None

        if self.network_type == "MLP":
            mlp_output_dims = (
                (*map_shape, 1)
                if self.trained_autoencoder is None
                else (self.trained_autoencoder.latent_dim,)
            )

        return {
            "version": 2,
            "sources": self.data.source_hashes,
            "split_ids": self.data.split_machine_ids,
            "map_axes": {
                name: values.tolist() for name, values in self.data.map_axes.items()
            },
            "target_minimum": self.data.target_minimum.tolist(),
            "target_denominator": self.data.target_denominator.tolist(),
            "feature_names": list(MACHINE_DEFINITION),
            "channel_names": list(OUTPUT_NEURON_LIST),
            "network": {
                "type": self.network_type,
                "channel_index": self.channel_index,
                "channel_name": OUTPUT_NEURON_LIST[self.channel_index],
                "map_shape": list(map_shape),
                "mlp_output_dims": (
                    list(mlp_output_dims) if mlp_output_dims is not None else None
                ),
                "fixed_autoencoder": self._autoencoder_config(),
            },
            "workers": self.num_workers,
            "training": {
                "learning_rate": self.trainer.learning_rate,
                "weight_decay": self.trainer.weight_decay,
                "epochs": self.trainer.num_epochs,
                "patience": self.trainer.patience,
                "loss_scope": (
                    "small_map"
                    if self.network_type == "MLP"
                    and self.trained_autoencoder is not None
                    else "full_map"
                ),
            },
            "search": asdict(self.search_space),
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
                self.cancellation.raise_if_requested()

                process = Process(
                    target=_optuna_worker,
                    kwargs={
                        "data": self.data,
                        "trainer": self.trainer,
                        "network_type": self.network_type,
                        "channel_index": self.channel_index,
                        "search_space": self.search_space,
                        "n_trials": self.n_trials,
                        "progress_queue": self.progress_queue,
                        "study_name": self.study_name,
                        "storage": self.storage,
                        "seed": self.seed + index,
                        "objective_seed": self.seed,
                        "log_queue": self.log_queue,
                        "trained_autoencoder": self.trained_autoencoder,
                    },
                )

                processes.append(process)
                process.start()

            while True:
                self.cancellation.raise_if_requested()

                for process in processes:
                    self.cancellation.raise_if_requested()
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
        network_type: NetworkType,
        channel_index: int,
        search_space: NetworkSearchSpace,
        seed: int,
        trained_autoencoder: Conv3dAutoencoder | None = None,
    ) -> None:
        if network_type not in ("MLP", "CNN"):
            raise ValueError("network_type must be 'MLP' or 'CNN'.")

        if type(channel_index) is not int or not 0 <= channel_index < len(
            OUTPUT_NEURON_LIST
        ):
            raise ValueError("Invalid output channel index.")

        if network_type == "CNN" and trained_autoencoder is not None:
            raise ValueError("CNN optimization cannot receive a trained autoencoder.")

        self.data: TrainValidationTestDataTensors = data
        self.trainer: NNTrainer = trainer
        self.network_type: NetworkType = network_type
        self.channel_index: int = channel_index
        self.search_space: NetworkSearchSpace = search_space
        self.seed: int = seed

        self.trained_autoencoder: Conv3dAutoencoder | None = copy.deepcopy(
            trained_autoencoder
        )

        map_shape: tuple[int, ...] = tuple(data.train.map_tensor.shape[1:-1])

        if self.trained_autoencoder is not None:
            if self.trained_autoencoder.input_channels != 1:
                raise ValueError("The supplied CNN must process one output channel.")

            if tuple(self.trained_autoencoder.map_shape) != map_shape:
                raise ValueError("The supplied CNN has an incompatible map shape.")

    def __call__(self, trial: optuna.Trial) -> float:
        trial_seed: int = (self.seed + trial.number) % 2**32
        random.seed(trial_seed)
        np.random.seed(trial_seed)
        torch.manual_seed(trial_seed)

        layers: tuple[int, ...] = self._get_layer(trial)
        latent_dim: int | None = None

        if self.network_type == "CNN":
            latent_dim = trial.suggest_int(
                "latent_dim",
                self.search_space.min_latent_dim,
                self.search_space.max_latent_dim,
                step=self.search_space.latent_step,
            )

        architecture: NetworkArchitecture = NetworkArchitecture(
            network_type=checked_network_type(self.network_type),
            layers=layers,
            latent_dim=latent_dim,
            input_channels=1,
        )

        train_data: MachineDataTensor = self.data.train.select_channel(
            self.channel_index
        )
        validation_data: MachineDataTensor = self.data.validation.select_channel(
            self.channel_index
        )
        map_shape: tuple[int, ...] = tuple(train_data.map_tensor.shape[1:-1])

        autoencoder: Conv3dAutoencoder | None = copy.deepcopy(self.trained_autoencoder)
        network: MLP | Conv3dAutoencoder

        if architecture.network_type == "CNN":
            if architecture.latent_dim is None:
                raise ValueError("A CNN requires a latent dimension.")

            network = Conv3dAutoencoder(
                input_channels=architecture.input_channels,
                features=architecture.layers,
                latent_dim=architecture.latent_dim,
                map_shape=map_shape,
            )
        else:
            output_dims: tuple[int, ...] = (
                (*map_shape, 1) if autoencoder is None else (autoencoder.latent_dim,)
            )

            network = MLP(
                num_input_neurons=train_data.geometry_tensor.shape[1],
                output_dims=output_dims,
                hidden_layer_neurons=architecture.layers,
            )

        self.trainer.start(
            neuronal_network=network,
            train_data=train_data,
            validation_data=validation_data,
            trained_autoencoder=autoencoder,
        )

        validation_loss: float = float(self.trainer.best_validation_loss)

        if not np.isfinite(validation_loss):
            raise FloatingPointError("The trial produced a non-finite validation loss.")

        trial.set_user_attr(
            "architecture",
            {
                "network_type": architecture.network_type,
                "layers": list(architecture.layers),
                "latent_dim": architecture.latent_dim,
                "input_channels": architecture.input_channels,
            },
        )
        trial.set_user_attr("channel_index", self.channel_index)
        trial.set_user_attr(
            "loss_scope",
            (
                "small_map"
                if architecture.network_type == "MLP" and autoencoder is not None
                else "full_map"
            ),
        )

        return validation_loss

    def _get_layer(self, trial: optuna.Trial) -> tuple[int, ...]:

        choices: tuple[int, ...] = self.search_space.unit_choices()

        if self.network_type == "MLP":
            count: int = trial.suggest_int(
                "n_layers",
                1,
                self.search_space.max_num_layers,
            )

            return tuple(
                int(
                    trial.suggest_categorical(
                        f"neurons_{index}",
                        choices,
                    )
                )
                for index in range(count)
            )

        layer_limit: int = min(
            self.search_space.max_num_layers,
            len(choices),
        )
        count = trial.suggest_int("n_layers", 1, layer_limit)

        features: list[int] = []
        previous_index: int = -1

        for layer_index in range(count):
            # Reserve sufficient larger feature counts for later layers.
            maximum_index: int = len(choices) - (count - layer_index)

            feature_index: int = trial.suggest_int(
                f"feature_index_{layer_index}",
                previous_index + 1,
                maximum_index,
            )

            features.append(choices[feature_index])
            previous_index = feature_index

        return tuple(features)
