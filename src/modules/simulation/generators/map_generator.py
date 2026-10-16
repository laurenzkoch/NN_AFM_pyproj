import numpy as np
from numpy.typing import NDArray

from src.config.config_mcad_parameters import (
    D_CURRENT_DENSITY,
    MAP_SIZE,
    Q_CURRENT_DENSITY,
    SHAFT_SPEED,
)


def map_generator(
    i_min_max: tuple[float, float] = SHAFT_SPEED,
    j_min_max: tuple[float, float] = Q_CURRENT_DENSITY,
    k_min_max: tuple[float, float] = D_CURRENT_DENSITY,
    map_size: tuple[int, int, int] = MAP_SIZE,
) -> NDArray[np.float32]:
    """Generate columns (speed, q, d) using sizes (speed, d, q)."""
    if len(map_size) != 3:
        raise ValueError("map_size must contain exactly three axis sizes.")

    if any(type(size) is not int or size < 1 for size in map_size):
        raise ValueError("Axis sizes must be positive integers.")

    values: list[list[float]] = []

    for speed in np.linspace(*i_min_max, map_size[0], dtype=float):
        for q_current in np.linspace(*j_min_max, map_size[2], dtype=float):
            for d_current in np.linspace(*k_min_max, map_size[1], dtype=float):
                values.append(
                    [
                        float(speed),
                        float(q_current),
                        float(d_current),
                    ]
                )

    return np.asarray(values, dtype=np.float32)


if __name__ == "__main__":
    op_samples = map_generator()
    print(op_samples)
