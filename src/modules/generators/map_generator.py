
import numpy as np
from typing import Generator


def map_generator(
        i_min_max: tuple[float | int, float| int],
        j_min_max: tuple[float | int, float| int],
        k_min_max: tuple[float | int, float| int],
        num_operation_points_per_dim: int
) -> Generator[np.ndarray, None, None]:

    for i in np.linspace(i_min_max[0], i_min_max[1], num_operation_points_per_dim):
        for j in np.linspace(j_min_max[0], j_min_max[1], num_operation_points_per_dim):
            for k in np.linspace(k_min_max[0], k_min_max[1], num_operation_points_per_dim):
                yield np.array([i, j, k], dtype=np.float32)