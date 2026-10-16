import math

from src.config.config_mcad_parameters import DEFAULT_VALUES, MAP_SIZE
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
    total_points: int = 0

    for num_base_designs, num_lam_designs in combinations:
        # Punktzahl für Generator v2.
        point_count: int = num_base_designs * num_lam_designs * math.prod(MAP_SIZE)
        total_points += point_count

        generator: AFMMachineGenerator = AFMMachineGenerator(
            num_base_designs=num_base_designs,
            num_lam_designs=num_lam_designs,
            num_data_points=point_count,
        )

        checked_points: int = 0
        rejected_points: int = 0

        for point in generator.get_generator(_type="v2"):
            checked_points += 1

            parameters: dict[str, float] = DEFAULT_VALUES.copy()
            parameters.update(point)

            parameter_names: list[str] = list(parameters)
            parameter_values: list[float] = list(parameters.values())

            if not test_boundary_conditions(
                parameter_names,
                parameter_values,
            ):
                rejected_points += 1
                print(f"Boundary check failed: {parameters}")

        print(
            f"Combination ({num_base_designs}, {num_lam_designs}): "
            f"{rejected_points} of {checked_points} points rejected."
        )

    print(f"Expected operating points: {total_points}")


if __name__ == "__main__":
    main()
