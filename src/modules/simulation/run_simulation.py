import logging
from queue import Queue
import numpy as np
from typing import Iterable, Generator
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import batched
import pandas as pd

from src.modules.data_managment.csv_io import import_data_from_csv, write_row_in_csv
from src.modules.process.motor_cad_processes import mcad_instance_handling
from src.config.config_paths import *
from src.config.config_mcad_parameters import *

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

        logger.info(
            "Winding: offset=%s coils=%s",
            mcad.get_value_from_mcad("MagWindingOffset"),
            mcad.get_value_from_mcad("NumberOfCoils"),
        )

        return mcad.execute_mcad_simulation()

    except Exception:
        mcad_instance_handling.shutdown_mcad_instance()
        raise


class MCADSimulation:

    def __init__(
            self,
            point_generator: Generator[dict[str, float], None, None],
            batch_size: int,
            progress_queue: Queue,
            csv_path: Path,
            num_workers: int,
            log_queue: Queue,
    ) -> None:
        self.point_generator = point_generator
        self.batch_size = batch_size
        self.progress_queue = progress_queue
        self.csv_path = csv_path
        self.num_workers = num_workers
        self.dataframes: dict[str, pd.DataFrame] = self._get_csv_files()
        self.log_queue = log_queue


    @staticmethod
    def _get_csv_files() -> dict[str, pd.DataFrame]:

        dataframes_dict = {}
        for csv_file in PATH_DATA_MCAD_RESULTS.glob("*.csv"):

            try:
                dataframe = import_data_from_csv(csv_file)
            except (
                    pd.errors.EmptyDataError,
                    pd.errors.ParserError,
                    UnicodeDecodeError,
            ) as exc:
                logger.warning(
                    "Skipping unreadable CSV %s: %s",
                    csv_file.name,
                    exc,
                )
                continue

            missing_columns = [
                column
                for column in CSV_PARAMETERS
                if column not in dataframe.columns
            ]

            if missing_columns:
                logger.warning(
                    "Skipping incompatible CSV %s. Missing columns: %s",
                    csv_file.name,
                    missing_columns,
                )
                continue

            dataframes_dict[csv_file.stem] = dataframe

        return dataframes_dict

    @staticmethod
    def _get_known_points(dataframe: pd.DataFrame, parameter_list: list[str]) -> set[tuple[int | float]]:
        known_points = {
            tuple(row)
            for row in dataframe[parameter_list].itertuples(
                index=True,
                name=None,
            )
        }
        return known_points


    def _get_all_known_points(self, parameter_list: list[str]) -> dict[str, set[tuple[int | float]]]:
        known_points: dict[str, set[tuple[int | float]]] = {}
        for key_csv_path, value_dataframe in self.dataframes.items():
            known_points[key_csv_path] = self._get_known_points(value_dataframe, parameter_list)
        return known_points


    def run_simulation(self) -> None:
        try:
            for batch in batched(
                    self.point_generator,
                    self.batch_size
            ):
                self._run_simulation_pool(batch)
            self.progress_queue.put(("DONE", None))
        except Exception as exc:
            self.progress_queue.put(("ERROR", exc))
            logger.exception(f"An error occurred during simulation: {exc}")
            raise


    def _run_simulation_pool(
            self,
            batch_machine_generator_instance: Iterable[dict[str, float]],
    ) -> None:

        with ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=mcad_instance_handling.init_mcad_instance,
            initargs=(self.log_queue,),
        ) as executor:

            futures = []
            for parameter_dict in batch_machine_generator_instance:

                parameter_list = list(parameter_dict.keys())
                known_points = self._get_all_known_points(parameter_list)
                point = tuple(parameter_dict.values())

                point_found = False

                for key_csv_path, known_points_set in known_points.items():
                    for known_point in known_points_set:
                        if all(
                                abs(a - b) <= 1e-4
                                for a, b in zip(point, known_point[1:])
                            ):
                            point_found = True
                            if key_csv_path == self.csv_path.stem:
                                self.progress_queue.put(("point existing", None))
                            else:
                                values_row: list[float] = (
                                    self.dataframes[key_csv_path]
                                    .loc[known_point[0], CSV_PARAMETERS]
                                    .tolist()
                                )
                                write_row_in_csv(self.csv_path, values_row)
                                self.progress_queue.put(("point copied", None))
                            break

                    if point_found:
                        break

                if point_found:
                    continue

                futures.append(
                    executor.submit(
                        _func,
                        parameter_dict=parameter_dict,
                    )
                )

            try:
                for future in as_completed(futures):
                    values_row, simulation_succeeded = future.result()
                    self.progress_queue.put(("progress", simulation_succeeded))

                    if simulation_succeeded and values_row is not None:
                        write_row_in_csv(self.csv_path, values_row)
            except Exception as exc:

                logger.error(f"Simulation pool failed: {exc!r}")

                for future in futures:
                    future.cancel()

                executor.shutdown(
                    wait=True,
                    cancel_futures=True,
                )

                raise
