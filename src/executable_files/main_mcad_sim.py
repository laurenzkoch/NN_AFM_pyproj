from src.modules.simulation.pipeline import SimulationPipeline
from src.modules.generators.machine_generator import AFMMachineGenerator


def main():

    pipeline = SimulationPipeline(
        num_base_designs=3,
        num_lam_designs=3,
        num_operation_points_per_dim=5,
        num_kernel=4,
        autostart=True,
        start_eduvpn=False
    )

    pipeline.start()
    pipeline.stop()



if __name__ == "__main__":
    main()