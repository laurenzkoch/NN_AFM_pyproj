import argparse
import json
import logging
from pathlib import Path
from typing import TypedDict

import optuna
import torch

from src.config.config_logging import configure_application_logging
from src.config.config_mcad_parameters import OUTPUT_NEURON_LIST
from src.config.config_paths import PATH_TRAINING_DATA
from src.modules.data_managment.experiment_utils import file_sha256
from src.modules.neuronal_networks.autoencoder_collection import (
    AutoencoderCollection,
)
from src.modules.neuronal_networks.model_data_io.autoencoder_io import (
    AutoencoderCollectionCheckpoint,
    load_autoencoders,
    save_autoencoders,
)
from src.modules.neuronal_networks.model_data_io.evaluation_io import (
    save_evaluation,
)
from src.modules.neuronal_networks.model_data_io.model_io import (
    JSONValue,
    load_bundle,
    save_bundle,
)
from src.modules.neuronal_networks.model_data_io.model_paths import (
    make_model_path,
)
from src.modules.neuronal_networks.optimization.optimization_pipline import (
    OptimizationPipline,
)
from src.modules.neuronal_networks.optimization.optimization_types import (
    NetworkSearchSpace,
    NetworkType,
)
from src.modules.neuronal_networks.surrogate_model import SurrogateModel
from src.modules.neuronal_networks.testing import evaluate_surrogate
from src.modules.plotting.prediction_report import create_prediction_report
from src.modules.plotting.prediction_results import plot_prediction_results

logger = logging.getLogger(__name__)


class OptimizationArguments(argparse.Namespace):
    max_layer: int
    max_neuron: int
    unit_step: int
    csv: str
    seed: int
    trials: int
    worker_timeout: float | None
    network_type: NetworkType
    autoencoders: Path | None


class TrainingConfig(TypedDict):
    learning_rate: float
    weight_decay: float
    num_epochs: int
    patience: int


def _search_metrics(
    pipeline: OptimizationPipline,
) -> list[JSONValue]:

    entries: list[JSONValue] = []
    for result in pipeline.optimization_results:
        layers: list[JSONValue] = [value for value in result.architecture.layers]

        entry: dict[str, JSONValue] = {
            "channel": OUTPUT_NEURON_LIST[result.channel_index],
            "network_type": result.architecture.network_type,
            "layers": layers,
            "latent_dim": result.architecture.latent_dim,
            "trial_number": result.trial_number,
            "validation_mse": result.mse,
        }
        entries.append(entry)

    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_layer", type=int, default=5)
    parser.add_argument("--max_neuron", type=int, default=500)
    parser.add_argument("--unit_step", type=int, default=10)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=4000)
    parser.add_argument("--worker-timeout", type=float, default=None)
    parser.add_argument(
        "--network-type",
        choices=("MLP", "CNN"),
        default="MLP",
    )
    parser.add_argument(
        "--autoencoders",
        type=Path,
        default=None,
    )

    args: OptimizationArguments = OptimizationArguments()
    parser.parse_args(namespace=args)

    if args.network_type == "CNN" and args.autoencoders is not None:
        parser.error("--autoencoders can only be combined with --network-type MLP.")

    configure_application_logging()
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    optuna.logging.disable_default_handler()
    optuna.logging.enable_propagation()

    autoencoders: AutoencoderCollection | None = None
    cnn_checkpoint: AutoencoderCollectionCheckpoint | None = None
    cnn_training_source_hash: str | None = None

    if args.autoencoders is not None:
        loaded_collection, loaded_checkpoint = load_autoencoders(args.autoencoders)

        training_source: JSONValue = loaded_checkpoint["metrics"].get("training_csv")

        if not isinstance(training_source, str):
            raise ValueError("The CNN collection does not identify its training CSV.")

        cnn_training_source_hash = loaded_collection.source_hashes.get(training_source)

        if cnn_training_source_hash is None:
            raise ValueError("The CNN training CSV has no saved source hash.")

        autoencoders = loaded_collection
        cnn_checkpoint = loaded_checkpoint

    training_csv_path: Path = (PATH_TRAINING_DATA / args.csv).resolve()

    test_sources: list[str] = []

    for path in sorted(PATH_TRAINING_DATA.iterdir()):
        if (
            not path.is_file()
            or path.suffix.lower() != ".csv"
            or path.resolve() == training_csv_path
        ):
            continue

        if (
            cnn_training_source_hash is not None
            and file_sha256(path) == cnn_training_source_hash
        ):
            logger.info(
                "Excluded CNN training source from independent tests: %s",
                path.name,
            )
            continue

        test_sources.append(path.name)

    test_filenames: tuple[str, ...] = tuple(test_sources)

    logger.info("Training/validation source: %s", args.csv)
    logger.info("Optimization type: %s", args.network_type)
    logger.info(
        "Automatically selected test sources (%d): %s",
        len(test_filenames),
        ", ".join(test_filenames) or "none",
    )

    training: TrainingConfig = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-3,
        "num_epochs": 2000,
        "patience": 100,
    }

    search_space: NetworkSearchSpace = NetworkSearchSpace(
        max_num_layers=args.max_layer,
        max_num_units_per_layer=args.max_neuron,
        unit_step=args.unit_step,
        min_latent_dim=1,
        max_latent_dim=50,
        latent_step=1,
    )

    pipeline: OptimizationPipline = OptimizationPipline(
        csv_file=args.csv,
        network_type=args.network_type,
        search_space=search_space,
        test_filenames=test_filenames,
        n_trials=args.trials,
        num_kernel=4,
        seed=args.seed,
        worker_timeout_seconds=args.worker_timeout,
        autoencoders=autoencoders,
        **training,
    )

    trained: SurrogateModel | AutoencoderCollection = pipeline.start()
    data = pipeline.data

    metrics: dict[str, JSONValue] = {
        "training_csv": args.csv,
        "search_trials": _search_metrics(pipeline),
        "training_config": {
            "learning_rate": training["learning_rate"],
            "weight_decay": training["weight_decay"],
            "num_epochs": training["num_epochs"],
            "patience": training["patience"],
        },
    }

    model_path: Path
    restored: SurrogateModel | AutoencoderCollection
    if isinstance(trained, AutoencoderCollection):
        model_path = make_model_path("AutoencoderCollections")

        save_autoencoders(
            path=model_path,
            collection=trained,
            metrics=metrics,
            seed=args.seed,
        )

        restored_collection, _ = load_autoencoders(model_path)

        with torch.no_grad():
            for channel_index, (original_cnn, restored_cnn) in enumerate(
                zip(
                    trained.autoencoders,
                    restored_collection.autoencoders,
                    strict=True,
                )
            ):
                reference_map: torch.Tensor = data.validation.map_tensor[
                    ..., channel_index : channel_index + 1
                ]

                torch.testing.assert_close(
                    original_cnn(reference_map),
                    restored_cnn(reference_map),
                )

        restored = restored_collection
    else:
        model_path = make_model_path(
            "Reference_model" if autoencoders is None else "Combi_model"
        )

        metrics["final_validation"] = evaluate_surrogate(
            trained,
            data.validation,
            data.target_denominator,
        )
        metrics["final_test"] = (
            evaluate_surrogate(
                trained,
                data.test,
                data.target_denominator,
            )
            if data.test is not None
            else None
        )

        if cnn_checkpoint is not None:
            metrics["autoencoder_training_config"] = cnn_checkpoint["metrics"].get(
                "training_config"
            )
            metrics["autoencoder_seed"] = cnn_checkpoint["seed"]

        save_bundle(
            path=model_path,
            model=trained,
            data=data,
            metrics=metrics,
            seed=args.seed,
            autoencoders=autoencoders,
        )

        restored_model, _ = load_bundle(model_path)

        torch.testing.assert_close(
            trained.get_prediction(data.validation.geometry_tensor),
            restored_model.get_prediction(data.validation.geometry_tensor),
        )

        restored = restored_model

    evaluation_path: Path = save_evaluation(
        model_path=model_path,
        model=restored,
        data=data,
    )

    plot_directory: Path = plot_prediction_results(model_path.parent)
    report_path: Path = create_prediction_report(model_path.parent)

    logger.info("Evaluation data saved: %s", evaluation_path)
    logger.info("Plots saved: %s", plot_directory)
    logger.info("Report saved: %s", report_path)

    logger.info(
        "Metrics:\n%s",
        json.dumps(
            metrics,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        ),
    )


if __name__ == "__main__":
    main()


# Referenzmodell
# python -m src.executable_files.main_optimize --csv "Results_EMagnetic_simulation_1_20_5_GenType-v2_26.10.2026_16-14-42_CET_99d7fc38.csv" --network-type MLP --trials 40 --max_layer 5 --max_neuron 500 --unit_step 10


# CNN-Sammlung
# python -m src.executable_files.main_optimize --csv "Results_EMagnetic_simulation_1_60_5_GenType-v2_19.10.2026_20-46-38_CEST_b3f8f82f.csv" --network-type CNN --trials 1000

# Kombimodell
# python -m src.executable_files.main_optimize --csv "DATEI.csv" --network-type
# MLP --autoencoders "VOLLSTAENDIGER_PFAD\AutoencoderCollection.pt"
