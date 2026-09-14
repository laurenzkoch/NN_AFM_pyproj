
import numpy as np

from src.config.config_mcad_parameters import *


def map_generator(
        i_min_max: tuple[float | int, float| int],
        j_min_max: tuple[float | int, float| int],
        k_min_max: tuple[float | int, float| int],
        num_operation_points_per_dim: int
) -> np.ndarray:

    values = []
    for i in np.linspace(i_min_max[0], i_min_max[1], num_operation_points_per_dim):
        for j in np.linspace(j_min_max[0], j_min_max[1], num_operation_points_per_dim):
            for k in np.linspace(k_min_max[0], k_min_max[1], num_operation_points_per_dim):
                values.append([i, j, k])
    return np.array(values, dtype=np.float32)



if __name__ == "__main__":
    op_samples = map_generator(
        i_min_max=SHAFT_SPEED,
        j_min_max=Q_CURRENT_DENSITY,
        k_min_max=D_CURRENT_DENSITY,
        num_operation_points_per_dim=5
    )
    print(op_samples)