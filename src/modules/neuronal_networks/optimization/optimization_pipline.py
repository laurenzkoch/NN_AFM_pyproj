import logging

from src.config.config_mcad_parameters import OUTPUT_NEURON_LIST
from src.config.config_paths import PATH_TRAINING_DATA
from src.modules.data_managment.experiment_utils import validate_search_settings
from src.modules.neuronal_networks.autoencoder_collection import (
    AutoencoderCollection,
)
from src.modules.neuronal_networks.manipulating_data import (
    DataLoader,
    TrainValidationTestDataTensors,
)
from src.modules.neuronal_networks.nn_training_torch import NNTrainer
from src.modules.neuronal_networks.optimization.optimization_service import (
    OptimizationService,
)
from src.modules.neuronal_networks.optimization.optimization_types import (
    ChannelOptimizationResult,
    NetworkSearchSpace,
    NetworkType,
)
from src.modules.neuronal_networks.surrogate_model import SurrogateModel
from src.modules.Pipeline_and_GUI.events import EventType, PipelineEvent
from src.modules.Pipeline_and_GUI.pipeline import MyPipeline

logger = logging.getLogger(__name__)


class OptimizationPipline(MyPipeline):
    def __init__(
        self,
        csv_file: str,
        network_type: NetworkType,
        search_space: NetworkSearchSpace,
        n_trials: int,
        num_kernel: int,
        learning_rate: float = 1e-3,
        patience: int = 50,
        num_epochs: int = 2000,
        weight_decay: float = 1e-3,
        test_filenames: tuple[str, ...] = (),
        seed: int = 42,
        worker_timeout_seconds: float | None = None,
        autoencoders: AutoencoderCollection | None = None,
    ) -> None:
        validate_search_settings(
            search_space.max_num_layers,
            search_space.max_num_units_per_layer,
            search_space.unit_step,
            n_trials,
            1,
        )

        if network_type not in ("MLP", "CNN"):
            raise ValueError("network_type must be 'MLP' or 'CNN'.")

        if network_type == "CNN" and autoencoders is not None:
            raise ValueError("CNN optimization cannot receive trained autoencoders.")

        trainer: NNTrainer = NNTrainer(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            patience=patience,
        )

        super().__init__(
            service_name="Optimization",
            num_kernel=num_kernel,
        )

        try:
            self.data: TrainValidationTestDataTensors = DataLoader(
                PATH_TRAINING_DATA
            ).load(
                filename=csv_file,
                test_filenames=tuple(test_filenames),
            )

            if autoencoders is not None:
                self.data = autoencoders.prepare_data(self.data)

            self.network_type: NetworkType = network_type
            self.trainer: NNTrainer = trainer
            self.completed_trials: int = 0
            self.status_text: str = "Ready"

            self.max_progress_steps: int = (
                len(OUTPUT_NEURON_LIST) * n_trials * self.num_workers
            )

            self.service: OptimizationService = OptimizationService(
                data=self.data,
                trainer=self.trainer,
                network_type=network_type,
                search_space=search_space,
                progress_queue=self.progress_queue,
                log_queue=self.log_queue,
                cancellation=self.cancellation,
                num_workers=self.num_workers,
                n_trials=n_trials,
                seed=seed,
                worker_timeout_seconds=worker_timeout_seconds,
                autoencoders=autoencoders,
            )

        except BaseException:
            try:
                self._close_queues()
            except Exception:
                logger.exception(
                    "Queue cleanup failed during optimization initialization"
                )
            raise

        self.optimization_results: tuple[ChannelOptimizationResult, ...] = ()

        self.result: SurrogateModel | AutoencoderCollection | None = None

    def run(self) -> SurrogateModel | AutoencoderCollection:
        return self.start()

    def start(self) -> SurrogateModel | AutoencoderCollection:
        self._claim_run()
        self._run_pipeline()

        result: SurrogateModel | AutoencoderCollection | None = self.service.result

        if result is None:
            raise RuntimeError("Optimization finished without trained networks.")

        self.optimization_results = self.service.optimization_results
        self.result = result

        return result

    def _queue_adapter(
        self,
        event: PipelineEvent,
    ) -> tuple[int, str, bool]:
        finished: bool = False

        if event.event_type == EventType.STARTED:
            self.status_text = event.text

        elif event.event_type == EventType.PROGRESS:
            if event.outcome == "trial_completed":
                self.completed_trials += event.increment

        elif event.event_type in (
            EventType.FINISHED,
            EventType.CANCELLED,
            EventType.FAILED,
        ):
            self.status_text = event.text
            finished = True

        return self.completed_trials, self.status_text, finished
