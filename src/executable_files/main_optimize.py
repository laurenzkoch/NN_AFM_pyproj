import multiprocessing

from src.config.config_logging import setup_main_logging
from src.modules.neuronal_networks.optimization.optimization_pipline import OptimizationPipline


def main():
    log_queue = multiprocessing.Queue()

    log_listener = setup_main_logging(log_queue)

    try:
        pipeline = OptimizationPipline(
            num_kernel=4,
            max_num_neuron_per_layer=200,
            max_num_layers=3,
            neuron_step=5,
            learning_rate=1e-3,
            patience=50,
            num_epochs=2000,
            weight_decay=1e-3,
        )

        pipeline.start()
        pipeline.stop()

    finally:
        log_listener.stop()


if __name__ == "__main__":
    main()
