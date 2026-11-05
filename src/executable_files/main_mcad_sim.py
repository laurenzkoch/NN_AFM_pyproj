from src.config.config_logging import configure_application_logging
from src.modules.gui.userinterface import MyUserInterface
from src.modules.simulation.sim_pipeline import SimulationPipeline


def main() -> None:
    configure_application_logging()

    combinations: list[tuple[int, int]] = [
        #(1, 10),
        #(1, 20),
        #(1, 30),
        #(1, 40),
        #(1, 50),
        #(1, 60),
        #(1, 70),
        #(1, 80),
        #(1, 90),
        (1, 100),
        (1, 200),
        (1, 300),
        (100, 1),
        (200, 1),
        (300, 1),
        (100, 100),
        (200, 200),
        (300, 300),
    ]

    pipeline = SimulationPipeline(
        design_combinations=combinations,
        start_eduvpn=False,
        gen_type="v2",
        num_workers=4,
        ui_factory=MyUserInterface,
    )

    pipeline.start()


if __name__ == "__main__":
    main()
