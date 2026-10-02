import multiprocessing

from src.config.config_logging import setup_main_logging
from src.modules.simulation.pipeline import SimulationPipeline

def main():

    log_queue = multiprocessing.Queue()

    log_listener = setup_main_logging(log_queue)

    try:
        pipeline = SimulationPipeline(
            num_base_designs=1,
            num_lam_designs=30,
            num_operation_points_per_dim=5,
            num_kernel=4,
            autostart=True,
            start_eduvpn=False,
            gen_type="v2",
            # csv_path="Results_EMagnetic_simulation_1_10_5_GenType-v2_30.09.2026_15-08-54_CEST.csv",
            log_queue=log_queue,
        )

        pipeline.start()
        pipeline.stop()

    finally:
        log_listener.stop()


if __name__ == "__main__":
    main()

