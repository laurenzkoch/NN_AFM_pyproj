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
        num_base_designs=1,
        num_lam_designs=10,
        num_operation_points_per_dim=5,
        num_kernel=4,
        autostart=True,
        start_eduvpn=False,
        gen_type="v2",
        #csv_path="Results_EMagnetic_simulation_2_3_5_GenType-v2_22.09.2026_19-36-34_CEST.csv",
    )

    pipeline.start()
    pipeline.stop()



if __name__ == "__main__":
        main()
