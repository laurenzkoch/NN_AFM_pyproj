from src.modules.simulation.pipeline import SimulationPipeline

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def main():

    pipeline = SimulationPipeline(
        num_base_designs=3,
        num_lam_designs=3,
        num_operation_points_per_dim=5,
        num_kernel=4,
        autostart=True,
        start_eduvpn=False,
        gen_type="v1",
        csv_path="Results_EMagnetic_simulation_3_3_5_18.09.2026_13-52-26_CEST.csv",
    )

    pipeline.start()
    pipeline.stop()



if __name__ == "__main__":
    main()





