
import numpy as np

from src.config.config_mcad_parameters import *


def map_generator(
        i_min_max: tuple[float | int, float| int] = SHAFT_SPEED,
        j_min_max: tuple[float | int, float| int] = Q_CURRENT_DENSITY,
        k_min_max: tuple[float | int, float| int] = D_CURRENT_DENSITY,
        num_operation_points_per_dim: int = 5,
) -> np.ndarray:

    values = []
    for i in np.linspace(i_min_max[0], i_min_max[1], num_operation_points_per_dim):
        for j in np.linspace(j_min_max[0], j_min_max[1], num_operation_points_per_dim):
            for k in np.linspace(k_min_max[0], k_min_max[1], num_operation_points_per_dim):
                values.append([i, j, k])
    return np.array(values, dtype=np.float32)



if __name__ == "__main__":
    op_samples = map_generator()
    print(op_samples)