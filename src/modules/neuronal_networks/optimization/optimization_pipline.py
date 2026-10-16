import logging
import math

from src.config.config_paths import PATH_TRAINING_DATA
from src.modules.data_managment.experiment_utils import validate_search_settings
from src.modules.neuronal_networks.manipulating_data import DataLoader
from src.modules.neuronal_networks.nn_training_torch import NNTrainer
from src.modules.neuronal_networks.optimization.optimizer import (
    NNOptimizer,
    OptimizationResult,
)
from src.modules.Pipeline_and_GUI.events import EventType, PipelineEvent
from src.modules.Pipeline_and_GUI.pipeline import MyPipeline

logger = logging.getLogger(__name__)


class OptimizationPipline(MyPipeline):
    def __init__(
        self,
        csv_file: str,
        max_num_neuron_per_layer: int,
        max_num_layers: int,
        neuron_step: int,
        n_trials: int,
        learning_rate: float = 1e-3,
        patience: int = 50,
        num_epochs: int = 2000,
        weight_decay: float = 1e-3,
        test_filenames: tuple[str, ...] = (),
        seed: int = 42,
        worker_timeout_seconds: float | None = None,
    ) -> None:
        validate_search_settings(
            max_num_layers, max_num_neuron_per_layer, neuron_step, n_trials, 1
        )

        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("Weight decay must be finite and nonnegative")

        super().__init__(
            service_name="Optimization",
            num_kernel=1,
        )

        try:
            self.data = DataLoader(PATH_TRAINING_DATA).load(
                filename=csv_file, test_filenames=tuple(test_filenames)
            )

            self.trained_models = 0
            self.status_text = "Ready"
            self.max_progress_steps = n_trials * self.num_workers
            self.num_workers = min(
                self.num_workers,
                self.max_progress_steps,
            )
            self.trainer = NNTrainer(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                num_epochs=num_epochs,
                patience=patience,
            )
            self.service: NNOptimizer = NNOptimizer(
                data=self.data,
                progress_queue=self.progress_queue,
                log_queue=self.log_queue,
                trainer=self.trainer,
                num_workers=self.num_workers,
                max_num_layers=max_num_layers,
                max_num_neurons_per_layer=max_num_neuron_per_layer,
                neuron_step=neuron_step,
                n_trials=n_trials,
                seed=seed,
                worker_timeout_seconds=worker_timeout_seconds,
            )
        except BaseException:
            try:
                self._close_queues()
            except Exception:
                logger.exception(
                    "Queue cleanup failed during optimization initialization"
                )
            raise

        self.optimization_result: OptimizationResult | None = None

    def run(self) -> OptimizationResult:
        return self.start()

    def start(self) -> OptimizationResult:
        self._claim_run()
        self._run_pipeline()

        result: OptimizationResult | None = self.service.result

        if result is None:
            raise RuntimeError("Optimization finished without a result")

        self.optimization_result = result
        return result

    def _queue_adapter(
        self,
        event: PipelineEvent,
    ) -> tuple[int, str, bool]:
        finished = False

        if event.event_type == EventType.STARTED:
            self.status_text = event.text

        elif event.event_type == EventType.PROGRESS:
            self.trained_models += event.increment
            self.status_text = "Running"

        elif event.event_type == EventType.FINISHED:
            self.status_text = event.text
            finished = True

        elif event.event_type == EventType.FAILED:
            self.status_text = event.text
            finished = True

        return self.trained_models, self.status_text, finished
