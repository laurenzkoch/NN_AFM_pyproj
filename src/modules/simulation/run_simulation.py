import numpy as np
from typing import Generator
from concurrent.futures import ProcessPoolExecutor, as_completed

from modules.data_managment.csv_io import *
from modules.process.resources import *
from modules.process.motor_cad_processes import mcad_instance_handling



def _func(
    parameter_dict: dict[str, float],
) -> tuple[list[float], bool]:

    mcad = mcad_instance_handling.get_mcad_instance()

    for name, value in parameter_dict.items():

        # np.float64, np.int64 etc. -> Python float/int
        if isinstance(value, np.generic):
            value = value.item()

        if name == "Pole_Number":
            tppc = mcad.get_value_from_mcad("TorquePointsPerCycle")
            mesh_points = value / 2 * tppc
            if mesh_points < 360:
                mesh_points = 4 * mesh_points
            mcad.set_value_in_mcad(variable_name="AirgapMeshPoints_mesh", value=mesh_points)
            mcad.set_value_in_mcad(variable_name="AirgapMeshPoints_layers", value=mesh_points)

        mcad.set_value_in_mcad(name, value)

    return mcad.execute_mcad_simulation()


def run_simulation_pool(
        machine_generator_instance: Generator[dict[str, float], None, None],
        progress_queue,
        csv_path: Path,
        num_workers: int,
) -> None:

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=mcad_instance_handling.init_mcad_instance
    ) as executor:

        futures = [
            executor.submit(
                _func,
                parameter_dict=parameter_dict,
            ) for parameter_dict in machine_generator_instance
        ]

        try:
            for future in as_completed(futures):
                all_values, simulation_succeeded = future.result()
                progress_queue.put(("progress", simulation_succeeded))

                if simulation_succeeded:
                    write_row_in_csv(csv_path, all_values)
        except Exception as exc:

            print(f"Simulation pool failed: {exc!r}")

            for future in futures:
                future.cancel()

            executor.terminate_workers()

            raise

        progress_queue.put(("DONE", None))










