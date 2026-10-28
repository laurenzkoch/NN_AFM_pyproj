from src.config.config_mcad_parameters import DEFAULT_VALUES
from src.modules.process.motor_cad_processes.mcad_processing import (
    test_boundary_conditions,
)
from src.modules.simulation.generators.machine_generator import (
    AFMMachineGenerator,
)


def main() -> None:
    combinations: tuple[tuple[int, int], ...] = (
        (1, 3),
        (1, 4),
        (1, 5),
    )

    total_geometries = 0

    for num_base_designs, num_lam_designs in combinations:
        expected_geometries = (
            num_base_designs * num_lam_designs
        )
        total_geometries += expected_geometries

        generator = AFMMachineGenerator(
            num_base_designs=num_base_designs,
            num_lam_designs=num_lam_designs,
        )

        checked_geometries = 0
        rejected_geometries = 0

        for geometry in generator.get_generator(_type="v2"):
            checked_geometries += 1

            parameters = {
                **DEFAULT_VALUES,
                **geometry,
                "PhaseAdvance": 0.0,
            }

            if not test_boundary_conditions(
                list(parameters),
                list(parameters.values()),
            ):
                rejected_geometries += 1
                print(f"Boundary check failed: {parameters}")

        if checked_geometries != expected_geometries:
            raise RuntimeError(
                f"Expected {expected_geometries} geometries, "
                f"received {checked_geometries}."
            )

        print(
            f"Combination ({num_base_designs}, {num_lam_designs}): "
            f"{rejected_geometries} of "
            f"{checked_geometries} geometries rejected."
        )

    print(f"Expected geometries: {total_geometries}")


if __name__ == "__main__":
    main()