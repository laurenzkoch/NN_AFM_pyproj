from src.config.config_logging import configure_application_logging
from src.modules.simulation.sim_pipeline import SimulationPipeline


def main() -> None:
    configure_application_logging()

    combinations: list[tuple[int, int]] = []
    for i in [1, 5, 10, 15]:
        for j in range(10, 300, 10):
            combinations.append((i, j))

    pipeline = SimulationPipeline(
        design_combinations=combinations,
        start_eduvpn=False,
        gen_type="v2",
        num_workers=4,
    )

    pipeline.start()


if __name__ == "__main__":
    main()
