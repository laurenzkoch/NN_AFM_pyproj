from src.config.config_logging import configure_application_logging
from src.modules.simulation.sim_pipeline import SimulationPipeline


def main() -> None:
    configure_application_logging()

    combinations: list[tuple[int, int]] = [
        (1, 10),
        (1, 20),
        (1, 30),
    ]

    pipeline = SimulationPipeline(
        design_combinations=combinations,
        start_eduvpn=False,
        gen_type="v2",
        num_workers=4,
    )

    pipeline.start()


if __name__ == "__main__":
    main()
