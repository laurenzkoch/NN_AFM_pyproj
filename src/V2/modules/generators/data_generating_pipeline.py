import numpy as np
from typing import Generator

from V2.modules.generators import machine_generator
from V2.config import config_paths, config_mcad_parameters
from V2.modules import userinterface
from V2.modules.utilitis import process_managing, data_managing
from V2.modules.motor_cad_processes import mcad_instance_handling

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from datetime import datetime
from zoneinfo import ZoneInfo




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
            mcad.set_value_in_mcad(variable_name="AirgapMeshPoints_mesh", value=value/2 * tppc)

        mcad.set_value_in_mcad(name, value)

    return mcad.execute_mcad_simulation()

class MCADDataGenerator:

    parameter_list = config_mcad_parameters.INPUT_NEURON_LIST

    def __init__(self):
        self.processmanager = process_managing.ProcessManager()
        self.datamanager = data_managing.DataManager()
        self.simulation_thread = None
        self.simulation_exception = None

    def __enter__(self) -> 'MCADDataGenerator':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.datamanager.delete_folder(config_paths.PATH_MCAD_DATA / config_paths.PATH_TEMP_FILES)
        return None


    def start(
        self,
        num_base_designs: int,
        num_sub_designs: int,
        num_subsub_designs: int,
        num_operation_points: int,
        num_kernel: int = 4,
    )-> None:

        number_of_datapoints = num_base_designs * num_sub_designs * num_subsub_designs * num_operation_points

        ui: userinterface.UserInterface | None = None

        def start_button_clicked() -> None:
            if ui is None:
                return

            ui.start_button_sim.config(state="disabled")

            self.generate_data(
                ui=ui,
                number_of_datapoints=number_of_datapoints,
                num_operation_points=num_operation_points,
                num_base_designs=num_base_designs,
                num_sub_designs=num_sub_designs,
                num_subsub_designs=num_subsub_designs,
                num_of_kernel=num_kernel,
            )

        ui = userinterface.UserInterface(
            max_value=number_of_datapoints,
            title="MCAD Simulation",
            button_function=start_button_clicked
        )

        ui.root.mainloop()

        if self.simulation_thread is not None:
            self.simulation_thread.join()

        if self.simulation_exception is not None:
            raise self.simulation_exception


    @staticmethod
    def create_csv_path(path_folder_results: Path, num_base_designs: int, num_sub_designs: int, num_subsub_designs: int) -> Path:

        current_time = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%d.%m.%Y_%H-%M-%S_%Z")
        path_folder = Path(path_folder_results)
        path_folder.mkdir(exist_ok=True)
        filename = f"Results_EMagnetic_simulation_{num_base_designs}_{num_sub_designs}_{num_subsub_designs}_{current_time}.csv"

        return path_folder / filename


    def generate_data(
            self,
            ui: userinterface.UserInterface,
            number_of_datapoints: int,
            num_operation_points: int,
            num_base_designs: int,
            num_sub_designs: int,
            num_subsub_designs: int,
            num_of_kernel: int = 4,
    ) -> None:

        generator = machine_generator.get_generator(
            num_operation_points=num_operation_points,
            num_base_designs=num_base_designs,
            num_sub_designs=num_sub_designs,
            num_subsub_designs=num_subsub_designs
        )

        self.simulation_thread = threading.Thread(
            target=self.run_simulation_pool,
            args=(
                generator,
                num_of_kernel,
                number_of_datapoints,
                num_base_designs,
                num_sub_designs,
                num_subsub_designs,
                ui.progress_queue
            ),
            daemon=False,
        )
        self.simulation_thread.start()


    def run_simulation_pool(
            self,
            machine_generator_instance: Generator[dict[str, float], None, None],
            num_of_kernels: int,
            num_of_datapoints: int,
            num_base_designs: int,
            num_sub_designs: int,
            num_subsub_designs: int,
            progress_queue,
    ) -> None:

        num_of_workers = min(
            self.processmanager.set_number_of_workers(number_of_kernels=num_of_kernels),
            num_of_datapoints
        )

        csv_path = self.create_csv_path(
            path_folder_results=config_paths.PATH_DATA_MCAD_RESULTS,
            num_base_designs=num_base_designs,
            num_sub_designs=num_sub_designs,
            num_subsub_designs=num_subsub_designs,
        )
        self.datamanager.create_csv_file(
            path=csv_path,
            columns_list=config_mcad_parameters.INPUT_NEURON_LIST + config_mcad_parameters.OUTPUT_NEURON_LIST,
        )

        progress_queue.put(("Starting MCAD", None))

        with ProcessPoolExecutor(
            max_workers=num_of_workers,
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
                    progress_queue.put(("Calculating...", None))

                    progress_queue.put(("progress", simulation_succeeded))
                    if simulation_succeeded:
                        self.datamanager.write_row_in_csv(csv_path, all_values)


            except Exception as exc:

                self.simulation_exception = exc
                print(
                    f"Simulation pool failed: {exc!r}",
                    flush=True
                )
                for future in futures:
                    future.cancel()
                executor.terminate_workers()

                return


            progress_queue.put(("DONE", None))