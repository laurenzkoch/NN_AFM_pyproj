import logging
import os

logger = logging.getLogger(__name__)


def determine_number_of_kernels(number_of_kernels: int = 4) -> int:
    if isinstance(number_of_kernels, bool) or not isinstance(number_of_kernels, int):
        raise TypeError("Number of kernels must be an integer.")

    if number_of_kernels < 1:
        raise ValueError("Number of kernels must be at least 1.")

    available_cpus: int | None = os.cpu_count()

    if available_cpus is None:
        logger.error("Could not determine the number of CPUs.")
        raise RuntimeError("Could not determine the number of CPUs.")

    if number_of_kernels > available_cpus:
        logger.warning(
            "Requested %d workers, but only %d CPUs are available. Using %d workers.",
            number_of_kernels,
            available_cpus,
            available_cpus,
        )

    return min(number_of_kernels, available_cpus)
