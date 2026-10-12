import logging
import os

logger = logging.getLogger(__name__)


def determine_number_of_kernels(number_of_kernels: int = 4) -> int:

    if number_of_kernels < 1:
        raise ValueError("Number of kernels must be at least 1")

    available_cpus = os.cpu_count()
    if available_cpus is None:
        logger.exception("Could not determine the number of CPUs.")
        raise RuntimeError("Could not determine the number of CPUs.")
    if number_of_kernels > available_cpus:
        logger.warning(f"There are not enough CPUs. Number of kernals is set all available CPUs: {available_cpus}.")

    number_of_kernels = min(number_of_kernels, available_cpus)

    return number_of_kernels
