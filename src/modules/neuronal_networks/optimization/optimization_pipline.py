import logging

from src.config.config_paths import PATH_TRAINING_DATA
from src.modules.Pipeline_and_GUI.events import PipelineEvent, EventType
from src.modules.Pipeline_and_GUI.pipeline import MyPipeline
from src.modules.neuronal_networks.manipulating_data import DataLoader, TrainValidationTestDataTensors
from src.modules.neuronal_networks.nn_training_torch import NNTrainer
from src.modules.neuronal_networks.optimization.optimizer import NNOptimizer, OptimizationResult

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
    ) -> None:

        dataloader = DataLoader(PATH_TRAINING_DATA)
        self.data: TrainValidationTestDataTensors = dataloader.load(csv_file)

        self.trained_models = 0
        self.status_text = "Ready"
        self.step: int = neuron_step
        self.max_num_neuron_per_layer: int = max_num_neuron_per_layer
        self.max_num_layers: int = max_num_layers

        self.optimization_result: OptimizationResult | None = None

        super().__init__(
            service_name="Optimization",
            num_kernel=1,
        )

        self.n_trials = n_trials
        self.max_progress_steps = self.n_trials * self.num_workers
        self.num_workers = min(
            self.num_workers,
            self.max_progress_steps,
        )

        self.trainer = NNTrainer(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            patience=patience
        )

        self.Service = NNOptimizer(
            data=self.data,
            progress_queue=self.progress_queue,
            trainer=self.trainer,
            num_workers=self.num_workers,
            max_num_layers=max_num_layers,
            max_num_neurons_per_layer=max_num_neuron_per_layer,
            neuron_step=neuron_step,
            n_trials=n_trials,
        )

    def start(self) -> OptimizationResult:
        self.run()
        assert isinstance(self.Service, NNOptimizer)
        result = self.Service.result

        if result is None:
            raise RuntimeError("Optimization finished without a result")

        return result


    def _calculate_datapoints(self) -> int:
        n = self.n_trials * self.num_workers
        return n

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
