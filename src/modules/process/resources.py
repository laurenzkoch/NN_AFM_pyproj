
import os


def determine_number_of_kernels(number_of_kernels: int = 4) -> int:

    available_cpus = os.cpu_count()
    if available_cpus is None:
        raise RuntimeError("Could not determine the number of CPUs.")
    if number_of_kernels > available_cpus:
        print(f"There are not enough CPUs. Number of kernals is set all available CPUs: {available_cpus}.")

    number_of_kernels = min(number_of_kernels, available_cpus)

    return number_of_kernels