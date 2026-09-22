import logging

import numpy as np
from typing import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.config.config_mcad_parameters import *
from src.modules.data_managment.csv_io import import_data_from_csv, write_row_in_csv
from src.modules.process.motor_cad_processes import mcad_instance_handling
from src.config.config_paths import *

logger = logging.getLogger(__name__)


def _func(
    parameter_dict: dict[str, float],
) -> tuple[list[float] | None, bool]:

    try:
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

    except Exception:
        mcad_instance_handling.shutdown_mcad_instance()
        raise


def run_simulation_pool(
        batch_machine_generator_instance: Iterable[dict[str, float]],
        progress_queue,
        csv_path: Path,
        num_workers: int,
) -> None:

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=mcad_instance_handling.init_mcad_instance
    ) as executor:

        data_df = import_data_from_csv(csv_path)

        batch = list(batch_machine_generator_instance)

        if not batch:
            return

        parameter_names = tuple(batch[0].keys())

        known_points = {
            tuple(row)
            for row in data_df[list(parameter_names)].itertuples(
                index=False,
                name=None,
            )
        }

        futures = []

        for parameter_dict in batch_machine_generator_instance:

            point = tuple(
                parameter_dict[name]
                for name in parameter_names
            )

            if any(
                    all(
                        abs(a - b) <= 1e-4
                        for a, b in zip(point, known_point)
                    )
                    for known_point in known_points
            ):
                progress_queue.put(("point existing", None))
                continue

            known_points.add(point)

            futures.append(
                executor.submit(
                    _func,
                    parameter_dict=parameter_dict,
                )
            )

        try:
            for future in as_completed(futures):
                values_row, simulation_succeeded = future.result()
                progress_queue.put(("progress", simulation_succeeded))

                if simulation_succeeded and values_row is not None:
                    write_row_in_csv(csv_path, values_row)
        except Exception as exc:

            logger.error(f"Simulation pool failed: {exc!r}")

            for future in futures:
                future.cancel()

            executor.shutdown(
                wait=True,
                cancel_futures=True,
            )

            raise
