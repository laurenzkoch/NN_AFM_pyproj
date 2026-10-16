import argparse
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

import numpy as np
import optuna
import torch

from src.config.config_logging import configure_application_logging
from src.config.config_mcad_parameters import MACHINE_DEFINITION, MAP_SIZE
from src.config.config_paths import PATH_DATA_TRAINED_NN
from src.modules.neuronal_networks.model_io import (
    JSONValue,
    load_bundle,
    save_bundle,
)
from src.modules.neuronal_networks.nn_training_torch import NNTrainer
from src.modules.neuronal_networks.NN_types.mlp_with_torch import MLP
from src.modules.neuronal_networks.optimization.optimization_pipline import (
    OptimizationPipline,
)
from src.modules.neuronal_networks.surrogate_model import SurrogateModelTrainer
from src.modules.neuronal_networks.testing import evaluate_surrogate

logger = logging.getLogger(__name__)


class OptimizationArguments(argparse.Namespace):
    csv: str
    test_csv: list[str]
    seed: int
    trials: int
    worker_timeout: float | None


class TrainingConfig(TypedDict):
    learning_rate: float
    weight_decay: float
    num_epochs: int
    patience: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--test-csv", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--worker-timeout", type=float, default=None)
    args: OptimizationArguments = OptimizationArguments()
    parser.parse_args(namespace=args)

    configure_application_logging()
    optuna.logging.disable_default_handler()
    optuna.logging.enable_propagation()

    training: TrainingConfig = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-3,
        "num_epochs": 2000,
        "patience": 50,
    }
    pipeline = OptimizationPipline(
        csv_file=args.csv,
        test_filenames=tuple(args.test_csv),
        max_num_neuron_per_layer=200,
        max_num_layers=3,
        neuron_step=5,
        n_trials=args.trials,
        seed=args.seed,
        worker_timeout_seconds=args.worker_timeout,
        **training,
    )
    search_result = pipeline.start()

    data = pipeline.data

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    template = MLP(len(MACHINE_DEFINITION), (*MAP_SIZE, 1), search_result.hidden_layers)
    final_model = SurrogateModelTrainer(NNTrainer(**training)).fit(
        mlp=template,
        train_data=data.train,
        validation_data=data.validation,
    )
    metrics: dict[str, JSONValue] = {
        "search_trial": {
            "number": search_result.trial_number,
            "mse": search_result.mse,
            "r2": search_result.r_squared,
        },
        "final_validation": evaluate_surrogate(
            final_model, data.validation, data.target_denominator
        ),
        "final_test": None,
        "training_config": {
            "learning_rate": training["learning_rate"],
            "weight_decay": training["weight_decay"],
            "num_epochs": training["num_epochs"],
            "patience": training["patience"],
        },
    }
    if data.test is not None:
        metrics["final_test"] = evaluate_surrogate(
            final_model, data.test, data.target_denominator
        )
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    model_path = Path(PATH_DATA_TRAINED_NN) / run_name / "surrogate.pt"
    save_bundle(
        model_path, final_model, search_result.hidden_layers, data, metrics, args.seed
    )
    loaded, _ = load_bundle(model_path)
    torch.testing.assert_close(
        final_model.get_prediction(data.validation.geometry_tensor),
        loaded.get_prediction(data.validation.geometry_tensor),
    )
    logger.info("Model saved: %s", model_path)
    logger.info("Metrics: %s", metrics)


if __name__ == "__main__":
    main()
