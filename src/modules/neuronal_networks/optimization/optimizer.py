import logging
from dataclasses import dataclass
from multiprocessing import Process, Queue

import optuna
import torch

from src.modules.Pipeline_and_GUI.events import PipelineEvent, EventType
from src.config.config_mcad_parameters import MACHINE_DEFINITION, MAP_SIZE
from src.modules.neuronal_networks.NN_types.mlp_with_torch import MLP
from src.modules.neuronal_networks.manipulating_data import TrainValidationTestDataTensors
from src.modules.neuronal_networks.nn_training_torch import NNTrainer
from src.modules.neuronal_networks.surrogate_model import SurrogateModel, SurrogateModelTrainer
from src.modules.neuronal_networks.testing import calc_mse, calc_r_squared

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationResult:
    hidden_layers: tuple[int, ...]
    mse: float | None
    r_squared: float
    trial_number: int


def _optuna_worker(
        data: TrainValidationTestDataTensors,
        trainer: NNTrainer,
        max_num_layers: int,
        max_num_neurons_per_layer: int,
        neuron_step: int,
        n_trials: int,
        progress_queue,
) -> None:
    torch.set_num_threads(1)

    study = optuna.create_study(
        study_name="layer_architecture",
        storage="sqlite:///optimization.db",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
    )

    objective = Objective(
        data=data,
        trainer=trainer,
        max_num_layers=max_num_layers,
        max_num_neurons_per_layer=max_num_neurons_per_layer,
        neuron_step=neuron_step,
    )

    def _progress_callback(_study, trial):
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


class NNOptimizer:

    def __init__(
            self,
            data: TrainValidationTestDataTensors,
            progress_queue: Queue,
            trainer: NNTrainer,
            max_num_layers: int,
            max_num_neurons_per_layer: int,
            neuron_step: int,
            num_workers: int = 4,
            n_trials: int = 50,
    ) -> None:

        self.progress_queue = progress_queue
        self.trainer = trainer
        self.max_num_layers = max_num_layers
        self.max_num_neurons_per_layer = max_num_neurons_per_layer
        self.step = neuron_step
        self.num_workers = num_workers
        self.n_trials = n_trials

        self.data: TrainValidationTestDataTensors = data

        self.result: OptimizationResult | None = None

    def run(self) -> None:
        self._run_workers()

        study = optuna.load_study(
            study_name="layer_architecture",
            storage="sqlite:///optimization.db",
        )

        best_trial = study.best_trial

        hidden_layers = tuple(
            best_trial.params[f"neurons_{i}"]
            for i in range(best_trial.params["n_layers"])
        )

        self.result = OptimizationResult(
            hidden_layers=hidden_layers,
            mse=best_trial.value,
            r_squared=best_trial.user_attrs["r2"],
            trial_number=best_trial.number,
        )


    def _run_workers(self) -> None:
        processes: list[Process] = []

        try:
            for _ in range(self.num_workers):
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
                    ),
                )

                # Vor start() aufnehmen, damit auch Startfehler
                # durch die Bereinigung abgedeckt sind.
                processes.append(process)
                process.start()

            for process in processes:
                process.join()

                if process.exitcode != 0:
                    raise RuntimeError(
                        f"Optuna worker {process.pid} failed "
                        f"with exit code {process.exitcode}"
                    )

        finally:
            self._cleanup_workers(processes)

    @staticmethod
    def _cleanup_workers(processes: list[Process]) -> None:
        started_processes = [
            process
            for process in processes
            if process.pid is not None
        ]

        for process in started_processes:
            if process.is_alive():
                process.terminate()

        for process in started_processes:
            process.join(timeout=5)

            if process.is_alive():
                logger.warning(
                    "Worker %s did not terminate; killing it",
                    process.pid,
                )
                process.kill()
                process.join()

class Objective:

    def __init__(
            self,
            data: TrainValidationTestDataTensors,
            trainer: NNTrainer,
            max_num_layers: int,
            max_num_neurons_per_layer: int,
            neuron_step: int,
            num_workers: int = 4,
    ):
        self.data = data
        self.max_num_layers = max_num_layers
        self.max_num_neurons_per_layer = max_num_neurons_per_layer
        self.step = neuron_step
        self.num_workers = num_workers
        self.trainer = trainer

    def __call__(self, trial):
        hidden_layer = self._get_layer(trial)

        mlp = MLP(
            num_input_neurons=len(MACHINE_DEFINITION),
            output_dims=(*MAP_SIZE, 1),
            hidden_layer_neurons=hidden_layer,
        )
        surrogatemodel_trainer = SurrogateModelTrainer(
            nn_trainer=self.trainer,
        )

        trained_surrogate_model: SurrogateModel = surrogatemodel_trainer.fit(
            mlp=mlp,
            train_data=self.data.train,
            validation_data=self.data.validation,
        )
        map_pred_vali = trained_surrogate_model.get_prediction(
            geo_tensor=self.data.validation.geometry_tensor,
        )
        mse = calc_mse(y_true=self.data.validation.map_tensor, y_pred=map_pred_vali)
        r2 = calc_r_squared(
            y_true=self.data.validation.map_tensor,
            y_pred=map_pred_vali,
        )

        trial.set_user_attr("r2", r2)

        return mse

    def _get_layer(self, trial):
        n_layers = trial.suggest_int(
            "n_layers",
            1,
            self.max_num_layers,
        )
        neuron_values = list(
            range(
                self.step,
                self.max_num_neurons_per_layer + self.step,
                self.step
            )
        )

        hidden_layers = []
        for i in range(n_layers):
            neurons = trial.suggest_categorical(
                f"neurons_{i}",
                neuron_values
            )
            hidden_layers.append(neurons)

        return tuple(hidden_layers)
