import numpy as np

from src.config.config_mcad_parameters import MAP_SIZE, SHAFT_SPEED, Q_CURRENT_DENSITY, D_CURRENT_DENSITY


def map_generator(
        i_min_max: tuple[float | int, float | int] = SHAFT_SPEED,
        j_min_max: tuple[float | int, float | int] = Q_CURRENT_DENSITY,
        k_min_max: tuple[float | int, float | int] = D_CURRENT_DENSITY,
        map_size: tuple[int, int, int] = MAP_SIZE,
) -> np.ndarray:
    values: list[list] = []
    for i in np.linspace(i_min_max[0], i_min_max[1], map_size[0], dtype=float):
        for j in np.linspace(j_min_max[0], j_min_max[1], map_size[1], dtype=float):
            for k in np.linspace(k_min_max[0], k_min_max[1], map_size[2], dtype=float):
                values.append([i, j, k])
    return np.array(values, dtype=np.float32)


if __name__ == "__main__":
    op_samples = map_generator()
    print(op_samples)
