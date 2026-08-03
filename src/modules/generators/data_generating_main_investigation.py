from typing import Generator

from modules.generators import machine_generator
from src.config import config_parameter_lists, config_paths
from src.modules import userinterface
from modules.utilitis import process_managing, data_managing
from modules.motor_cad_processes import mcad_instance_handling

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from datetime import datetime
from zoneinfo import ZoneInfo



def _func(
    parameter_dict: dict[str, float],
) -> tuple[list[float], bool]:

    mcad = mcad_instance_handling.get_mcad_instance()
    mcad_instance_handling.open_mcad_file(mcad)

    mcad.set_value_in_mcad(variable_name="MagPhases", value=3) #MagPhases = 3

    for name, value in parameter_dict.items():

        if name == "Pole_Number":
            tppc = mcad.get_value_from_mcad("TorquePointsPerCycle")
            mcad.set_value_in_mcad(variable_name="AirgapMeshPoints_layers", value=value * 4 * tppc)
            mcad.set_value_in_mcad(variable_name="AirgapMeshPoints_mesh", value=value * 4 * tppc)

        mcad.set_value_in_mcad(name, value)

    return mcad.execute_mcad_simulation()


class MCADDataGenerator:

    parameter_list = config_parameter_lists.INPUT_NEURON_LIST

    def __init__(self):
        self.processmanager = process_managing.ProcessManager()
        self.datamanager = data_managing.DataManager()

    def __enter__(self) -> 'MCADDataGenerator':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.datamanager.delete_folder(config_paths.PATH_MCAD_DATA / config_paths.PATH_TEMP_FILES)
        return None


    def start(
        self,
        variation_type: str,
        number_of_datapoints: int = 1000,
        num_kernel: int = 4,
    )-> None:

        ui: userinterface.UserInterface | None = None

        def start_button_clicked() -> None:
            if ui is None:
                return

            ui.start_button_sim.config(state="disabled")

            self.generate_data(
                ui=ui,
                variation_type=variation_type,
                number_of_datapoints=number_of_datapoints,
                num_kernel=num_kernel,
            )

        ui = userinterface.UserInterface(
            max_value=number_of_datapoints,
            title=f"MCAD Simulation: {variation_type} Variation",
            button_function=start_button_clicked
        )

        ui.root.mainloop()


    @staticmethod
    def create_csv_path(path_folder_results: Path, variation_type: str) -> Path:

        current_time = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%d.%m.%Y_%H-%M-%S_%Z")
        path_folder = Path(path_folder_results)
        path_folder.mkdir(exist_ok=True)
        filename = f"Results_EMagnetic_simulation_{variation_type}-variation_{current_time}.csv"

        return path_folder / filename


    def generate_data(
            self,
            ui: userinterface.UserInterface,
            variation_type: str,
            number_of_datapoints: int,
            num_kernel: int = 4,
    ) -> None:


        num_of_workers = min(
            self.processmanager.set_number_of_workers(number_of_kernels=num_kernel),
            number_of_datapoints
        )
        csv_path = self.create_csv_path(
            path_folder_results=config_paths.PATH_DATA_MCAD_RESULTS,
            variation_type=variation_type
        )
        self.datamanager.create_csv_file(
            path=csv_path,
            columns_list=config_parameter_lists.INPUT_NEURON_LIST + config_parameter_lists.OUTPUT_NEURON_LIST
        )

        match variation_type:
            case "Winding":
                generator = machine_generator.do_winding_variation(number_of_datapoints=number_of_datapoints)
            case "Axial":
                generator = machine_generator.do_axial_variation(number_of_datapoints=number_of_datapoints)
            case "Operational Point":
                generator = machine_generator.do_operation_point_variation(number_of_datapoints=number_of_datapoints)
            case "Radial":
                generator = machine_generator.do_radial_variation(number_of_datapoints=number_of_datapoints, slot_number=18)
            case _:
                raise ValueError(f"Variation type '{variation_type}' is not supported.")


        simulation_thread = threading.Thread(
            target=self.run_simulation_pool,
            args=(
                generator,
                csv_path,
                num_of_workers,
                ui.progress_queue,
            ),
            daemon=True,
        )
        simulation_thread.start()


    def run_simulation_pool(
            self,
            machine_generator_instance: Generator[dict[str, float], None, None],
            csv_path,
            num_of_workers,
            progress_queue,
    ) -> None:


        progress_queue.put(("Starting MCAD", None))

        with ProcessPoolExecutor(
            max_workers=num_of_workers,
            initializer=mcad_instance_handling.init_mcad_instance
        ) as executor:

            progress_queue.put(("Calculating...", None))

            futures = [
                executor.submit(
                    _func,
                    parameter_dict=parameter_dict,
                ) for parameter_dict in machine_generator_instance
            ]

            for future in as_completed(futures):
                all_values, simulation_succeeded = future.result()
                progress_queue.put(("progress", simulation_succeeded))

                if simulation_succeeded:
                    self.datamanager.write_row_in_csv(csv_path, all_values)

            progress_queue.put(("DONE", None))