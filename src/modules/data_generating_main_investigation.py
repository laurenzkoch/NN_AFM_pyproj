import atexit
import math
import subprocess
from itertools import product
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

from src.config import config_parameter_lists, config_paths
from src.config.config_paths import PATH_MCAD_BASEFILE
from src.modules import process_managing, data_managing, mcad_processing, userinterface, eduvpn_processing


_worker_mcad: mcad_processing.MCADProcessor | None = None
_worker_base_file: Path | None = None


def _init_worker(datamanager: data_managing.DataManager) -> None:
    global _worker_mcad, _worker_base_file

    atexit.register(_shutdown_worker)

    if _worker_mcad is None:
        _worker_mcad = mcad_processing.MCADProcessor()
        _worker_base_file = datamanager.create_temp_file_copy(
            Path(PATH_MCAD_BASEFILE)
        )
        if _worker_base_file is not None and _worker_mcad is not None:
            _worker_mcad.open_motorcad_file(file_path=_worker_base_file)


def _get_worker_mcad() -> mcad_processing.MCADProcessor:
    global _worker_mcad
    if _worker_mcad is None:
        raise RuntimeError("Worker MCAD instance not initialized")
    return _worker_mcad


def _shutdown_worker() -> None:
    global _worker_mcad
    if _worker_mcad is not None:
        _worker_mcad.__exit__(None, None, None)
        _worker_mcad = None


def _func(
    value_tuple: tuple[float, ...],
    parameter_names: tuple[str, ...],
    datamanager: data_managing.DataManager,
    csv_path_results: Path,
    normalize_data: bool = True,
) -> bool:

    mcad = _get_worker_mcad()

    mcad.set_value_in_mcad(variable_name="MagPhases", value=3) #MagPhases = 3
    mcad.set_value_in_mcad(variable_name="Shaft_Hole_Diameter", value=0) #Shaft is solid


    for value, name  in zip(value_tuple, parameter_names):
        if normalize_data:
            if name == parameter_names[0]:
                mcad.set_value_in_mcad(name, value)
            else:
                mcad.set_value_in_mcad(name, value_tuple[0] * value)
        else:
            if name == "Slot_Number":
                mcad.set_value_in_mcad(name, value * mcad.get_value_from_mcad("MagPhases"))
            elif name == "Pole_Number":
                mcad.set_value_in_mcad(name, int(value) * 2)
                tppc = mcad.get_value_from_mcad("TorquePointsPerCycle")
                mcad.set_value_in_mcad(variable_name="AirgapMeshPoints_layers", value=int(value) * 2 * tppc)
                mcad.set_value_in_mcad(variable_name="AirgapMeshPoints_mesh", value=int(value) * 2 * tppc)
            elif name == "Slot_Width":
                mcad.set_value_in_mcad(name, value * math.pi * mcad.get_value_from_mcad(
                    "AFM_D_Rotor") / mcad.get_value_from_mcad("Slot_Number"))
            elif name == "Slot_Depth":
                mcad.set_value_in_mcad(name, value * mcad.get_value_from_mcad("Stator_Lam_Length"))
            else:
                mcad.set_value_in_mcad(name, value)


    afm_d_rotor = mcad.get_value_from_mcad("AFM_D_Rotor")
    shaft_dia = mcad.get_value_from_mcad("Shaft_Dia")
    mcad.set_value_in_mcad(variable_name="Stator_Lam_Dia", value=afm_d_rotor) #Stator_Lam_Dia = AFM_D_Rotor
    mcad.set_value_in_mcad(variable_name="Stator_Bore", value=shaft_dia + 0.5) #Stator_Bore = Shaft_Dia + 1


    all_values, sim_suceed = mcad.execute_mcad_simulation()
    if sim_suceed:
        datamanager.write_row_in_csv(csv_path_results, all_values)
        return True
    else:
       return False



class MCADDataGenerator:

    parameter_list = config_parameter_lists.INPUT_NEURON_LIST

    def __init__(self):
        self.processmanager = process_managing.ProcessManager()
        self.datamanager = data_managing.DataManager()

    def __enter__(self) -> 'MCADDataGenerator':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None


    def start(
        self,
        parameter_dict: dict[str, list[float]],
        variation_type: str,
        number_of_datapoints: int = 1000,
        num_kernel: int = 4,
        normalize_data: bool = True,
    )-> None:

        parameter_names = tuple(parameter_dict.keys())
        value_list = self.generate_value_list(parameter_dict, number_of_datapoints)

        max_progress_steps = len(value_list)


        ui: userinterface.UserInterface | None = None

        def start_button_clicked() -> None:
            if ui is None:
                return

            ui.start_button.config(state="disabled")

            self.generate_data(
                ui=ui,
                parameter_names=parameter_names,
                variation_type=variation_type,
                max_progress_steps=max_progress_steps,
                value_list=value_list,
                num_kernel=num_kernel,
                normalize_data=normalize_data
            )

        ui = userinterface.UserInterface(
            max_value=max_progress_steps,
            title=f"MCAD Simulation {variation_type} Variation",
            button_function=start_button_clicked
        )

        if eduvpn_process is not None and ui is not None:
            ui.progress_queue.put(("EduVPN started", None))

        ui.root.mainloop()

    @staticmethod
    def generate_value_list(parameter_dict: dict[str, list[float]], number_of_datapoints: int) -> list[tuple[float, float, float]]:

        points_per_parameter = number_of_datapoints ** (1 / len(parameter_dict))

        for parameter_name, parameter_values in parameter_dict.items():
            parameter_dict[parameter_name] = np.arange(
                parameter_values[0],
                parameter_values[1],
                (parameter_values[1] - parameter_values[0]) / points_per_parameter
            ).tolist()
        value_list = list(product(*parameter_dict.values()))

        return value_list


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
            parameter_names: tuple[str, ...],
            variation_type: str,
            max_progress_steps: int,
            value_list: list[tuple[float, ...]],
            num_kernel: int = 4,
            normalize_data: bool = True,
    ) -> None:


        num_of_workers = min(
            self.processmanager.set_number_of_workers(number_of_kernels=num_kernel),
            max_progress_steps
        )
        csv_path = self.create_csv_path(
            path_folder_results=config_paths.PATH_DATA_MCAD_RESULTS,
            variation_type=variation_type
        )
        self.datamanager.create_csv_file(
            path=csv_path,
            columns_list=config_parameter_lists.INPUT_NEURON_LIST + config_parameter_lists.OUTPUT_NEURON_LIST
        )

        simulation_thread = threading.Thread(
            target=self.run_simulation_pool,
            args=(
                value_list,
                parameter_names,
                csv_path,
                num_of_workers,
                ui.progress_queue,
                normalize_data,
            ),
            daemon=True,
        )
        simulation_thread.start()




    def run_simulation_pool(
            self,
            value_list,
            parameter_names,
            csv_path,
            num_of_workers,
            progress_queue,
            normalize_data,
    ) -> None:

        progress_queue.put(("Starting MCAD", None))

        with ProcessPoolExecutor(
            max_workers=num_of_workers,
            initializer=_init_worker,
            initargs=(self.datamanager,)
        ) as executor:

            progress_queue.put(("Calculating...", None))

            futures = [
                executor.submit(
                    _func,
                    value_tuple,
                    parameter_names,
                    self.datamanager,
                    csv_path,
                    normalize_data=normalize_data
                ) for value_tuple in value_list
            ]

            for future in as_completed(futures):
                simulation_succeeded = future.result()
                progress_queue.put(("progress", simulation_succeeded))

            progress_queue.put(("DONE", None))

            self.datamanager.delete_folder(config_paths.PATH_MCAD_DATA / config_paths.PATH_TEMP_FILES)

