import logging
from dataclasses import dataclass, field
from multiprocessing.queues import Queue as ProcessQueue

import torch

from src.config.config_mcad_parameters import OUTPUT_NEURON_LIST
from src.modules.neuronal_networks.autoencoder_collection import (
    AutoencoderCollection,
)
from src.modules.neuronal_networks.autoencoder_collection_trainer import (
    AutoencoderCollectionTrainer,
)
from src.modules.neuronal_networks.manipulating_data import (
    TrainValidationTestDataTensors,
)
from src.modules.neuronal_networks.nn_training_torch import NNTrainer
from src.modules.neuronal_networks.NN_types.cnn_autoencoder import (
    Conv3dAutoencoder,
)
from src.modules.neuronal_networks.NN_types.mlp_with_torch import MLP
from src.modules.neuronal_networks.optimization.optimization_types import (
    ChannelOptimizationResult,
    NetworkSearchSpace,
    NetworkType,
)
from src.modules.neuronal_networks.optimization.optimizer import NNOptimizer
from src.modules.neuronal_networks.surrogate_model import (
    SubSurrogateModel,
    SurrogateModel,
    SurrogateModelTrainer,
)
from src.modules.Pipeline_and_GUI.cancellation import CancellationToken
from src.modules.Pipeline_and_GUI.events import EventType, PipelineEvent

logger = logging.getLogger(__name__)


@dataclass
class OptimizationService:
    data: TrainValidationTestDataTensors
    trainer: NNTrainer
    network_type: NetworkType
    search_space: NetworkSearchSpace
    progress_queue: ProcessQueue[PipelineEvent]
    log_queue: ProcessQueue[logging.LogRecord] | None
    cancellation: CancellationToken
    num_workers: int
    n_trials: int
    seed: int
    worker_timeout_seconds: float | None = None
    autoencoders: AutoencoderCollection | None = None

    optimization_results: tuple[ChannelOptimizationResult, ...] = field(
        default=(),
        init=False,
    )
    result: SurrogateModel | AutoencoderCollection | None = field(
        default=None,
        init=False,
    )

    def run(self) -> None:
        if self.network_type not in ("MLP", "CNN"):
            raise ValueError("network_type must be 'MLP' or 'CNN'.")

        if self.network_type == "CNN" and self.autoencoders is not None:
            raise ValueError("CNN optimization cannot receive trained autoencoders.")

        if self.autoencoders is not None:
            self.autoencoders.validate_compatible(self.data)

        results: list[ChannelOptimizationResult] = []

        for channel_index, channel_name in enumerate(OUTPUT_NEURON_LIST):
            self.cancellation.raise_if_requested()

            self.progress_queue.put(
                PipelineEvent(
                    event_type=EventType.STARTED,
                    text=(
                        f"Optimierung {channel_index + 1}/"
                        f"{len(OUTPUT_NEURON_LIST)}: {channel_name}"
                    ),
                )
            )

            channel_autoencoder: Conv3dAutoencoder | None = None

            if self.autoencoders is not None:
                channel_autoencoder = self.autoencoders.for_channel(channel_name)

            optimizer: NNOptimizer = NNOptimizer(
                data=self.data,
                progress_queue=self.progress_queue,
                trainer=self.trainer,
                network_type=self.network_type,
                channel_index=channel_index,
                search_space=self.search_space,
                num_workers=self.num_workers,
                n_trials=self.n_trials,
                seed=self.seed,
                worker_timeout_seconds=self.worker_timeout_seconds,
                log_queue=self.log_queue,
                cancellation=self.cancellation,
                trained_autoencoder=channel_autoencoder,
            )
            optimizer.run()

            channel_result: ChannelOptimizationResult | None = optimizer.channel_result

            if channel_result is None:
                raise RuntimeError(
                    f"No optimization result for channel: {channel_name}"
                )

            results.append(channel_result)

        self.optimization_results = tuple(results)
        self.cancellation.raise_if_requested()

        self.progress_queue.put(
            PipelineEvent(
                event_type=EventType.STARTED,
                text="Final training of the selected networks",
            )
        )

        trained_result: SurrogateModel | AutoencoderCollection = (
            self._train_selected_networks()
        )

        self.cancellation.raise_if_requested()
        self.result = trained_result

        logger.info(
            "Completed %s optimization and training for %d channels",
            self.network_type,
            len(self.optimization_results),
        )

    def _train_selected_networks(
        self,
    ) -> SurrogateModel | AutoencoderCollection:
        map_shape: tuple[int, ...] = tuple(self.data.train.map_tensor.shape[1:-1])
        input_count: int = self.data.train.geometry_tensor.shape[1]

        mlp_parts: list[SubSurrogateModel] = []
        cnn_models: list[Conv3dAutoencoder] = []

        for channel_result in self.optimization_results:
            self.cancellation.raise_if_requested()

            architecture = channel_result.architecture
            channel_name: str = OUTPUT_NEURON_LIST[channel_result.channel_index]

            # Reproduce the initialization used by the selected trial.
            torch.manual_seed((self.seed + channel_result.trial_number) % 2**32)

            if architecture.network_type == "CNN":
                latent_dim: int | None = architecture.latent_dim

                if latent_dim is None:
                    raise ValueError(f"Missing latent dimension: {channel_name}")

                cnn_models.append(
                    Conv3dAutoencoder(
                        input_channels=architecture.input_channels,
                        features=architecture.layers,
                        latent_dim=latent_dim,
                        map_shape=map_shape,
                    )
                )
                continue

            output_dims: tuple[int, ...] = (*map_shape, 1)

            if self.autoencoders is not None:
                output_dims = (self.autoencoders.for_channel(channel_name).latent_dim,)

            mlp_parts.append(
                SubSurrogateModel(
                    mlp=MLP(
                        num_input_neurons=input_count,
                        hidden_layer_neurons=architecture.layers,
                        output_dims=output_dims,
                    )
                )
            )

        if self.network_type == "CNN":
            return AutoencoderCollectionTrainer(self.trainer).fit(
                autoencoders=tuple(cnn_models),
                data=self.data,
            )

        model: SurrogateModel = SurrogateModel(mlp_parts)

        return SurrogateModelTrainer(self.trainer).fit(
            model=model,
            data=self.data,
            autoencoders=self.autoencoders,
        )
