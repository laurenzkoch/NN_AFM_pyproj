"""Generate shared operating-point coordinates for AFM simulations."""

from itertools import product

import numpy as np
from numpy.typing import NDArray

from src.config.config_mcad_parameters import (
    D_CURRENT_DENSITY,
    MAP_SIZE,
    Q_CURRENT_DENSITY,
    SHAFT_SPEED_NORM,
)


def map_generator(
    i_min_max: tuple[float, float] = SHAFT_SPEED_NORM,
    j_min_max: tuple[float, float] = Q_CURRENT_DENSITY,
    k_min_max: tuple[float, float] = D_CURRENT_DENSITY,
    map_size: tuple[int, int, int] = MAP_SIZE,
) -> NDArray[np.float64]:
    """Return rows of [normalized speed, q density, d density].

    Normalized speed is shaft_speed_rpm / base_speed_rpm.
    Current densities are RMS quantities in A/mm².

    The d-axis varies fastest, followed by q-axis and speed.
    """
    if len(map_size) != 3:
        raise ValueError("map_size must contain exactly three axis sizes.")

    if any(type(size) is not int or size < 1 for size in map_size):
        raise ValueError("Axis sizes must be positive integers.")

    bounds = np.asarray(
        [i_min_max, j_min_max, k_min_max],
        dtype=np.float64,
    )

    if bounds.shape != (3, 2):
        raise ValueError("Each axis must have exactly two bounds.")

    if not np.isfinite(bounds).all():
        raise ValueError("Axis bounds must be finite.")

    if (bounds[:, 0] > bounds[:, 1]).any():
        raise ValueError("Lower bounds must not exceed upper bounds.")

    if bounds[0, 0] < 0:
        raise ValueError("Normalized speed must not be negative.")

    for axis_index, size in enumerate(map_size):
        if size > 1 and bounds[axis_index, 0] == bounds[axis_index, 1]:
            raise ValueError(
                f"Axis {axis_index} needs distinct bounds when its size exceeds one."
            )

    axes = [
        np.linspace(lower, upper, size, dtype=np.float64)
        for (lower, upper), size in zip(bounds, map_size, strict=True)
    ]

    return np.asarray(
        list(product(*axes)),
        dtype=np.float64,
    )


if __name__ == "__main__":
    print(map_generator())
