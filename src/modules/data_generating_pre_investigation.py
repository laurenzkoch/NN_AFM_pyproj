from dataclasses import dataclass, field
from itertools import product
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import tempfile
import numpy as np
import pandas as pd

from modules.data_managing import DataManager
from src.modules import process_managing, data_managing
from src.config import config_paths, config_parameter_lists, config_process
from src.modules import mcad_processing


@dataclass
class PARAMETERDICT:
    variable_name_1: str = ""
    variable_name_2: str = "default"
    variable_name_3: str = "default"
    variable_config_1: tuple[float, float, float] = (0.0, 0.0, 0.0)
    variable_config_2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    variable_config_3: tuple[float, float, float] = (0.0, 0.0, 0.0)
    value_list_1: list[float] = field(default_factory=list)
    value_list_2: list[float] = field(default_factory=list)
    value_list_3: list[float] = field(default_factory=list)


class MCADDataGenerator:

    parameter_list = config_parameter_lists.INPUT_NEURON_LIST

    def __init__(self):
        self.processmanager = process_managing.ProcessManager()
        self.datamanager = data_managing.DataManager()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False



    @staticmethod
    def create_csv_path(path_folder_results: Path, parameter_dict: PARAMETERDICT, variation_type: bool = True) -> Path:

        current_time = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%d.%m.%Y_%H-%M-%S_%Z")

        if variation_type:
            filename_suffix = parameter_dict.variable_name_1
            path_folder = Path(path_folder_results / "single_parameter_variation")
        else:
            filename_suffix = parameter_dict.variable_name_1 + "_" + parameter_dict.variable_name_2
            path_folder = Path(path_folder_results / "multi_parameter_variation")
        path_folder.mkdir(exist_ok=True)

        filename = f"Results_EMagnetic_simulation_{filename_suffix}_{current_time}.csv"

        return path_folder / filename

    @staticmethod
    def execute_mcad_simulation(
            mcad_processor: mcad_processing.MCADProcessor,
            datamanager: DataManager,
            input_parameters: list[str],
            output_parameters: list[str],
            path_results: Path,
    ) -> None:

        def get_values(var_list: list[str]) -> list[float]:
            return [mcad_processor.get_value_from_mcad(var) for var in var_list]

        value_row = get_values(input_parameters)

        if mcad_processor.test_boundary_conditions(input_parameters, value_row):
            if mcad_processor.do_calculation():
                value_row.extend(
                    get_values(output_parameters)
                )
                datamanager.write_row_in_csv(path_results, value_row)
            else:
                for _var in input_parameters: print(f"{_var}: {value_row[input_parameters.index(_var)]}")


    @staticmethod
    def func(
            chunk: list[tuple[float, float, float]],
            parameter: PARAMETERDICT,
            input_parameter: list[str],
            output_parameters: list[str],
            temp_folder_path: Path,
            base_filename: str = "base_first_data.mot",
            datamanager: DataManager = DataManager(),
    ) -> None:

        with mcad_processing.MCADProcessor() as mcad:

            copy_base_file = datamanager.create_temp_file_copy(Path(config_paths.PATH_MCAD_DATA / base_filename))
            mcad.open_motorcad_file(file_path=copy_base_file)

            with tempfile.NamedTemporaryFile(
                    dir=temp_folder_path,
                    prefix=f"temp_",
                    suffix=".csv",
                    delete=False
            ) as temp_file:
                temp_csv_path = Path(temp_file.name)



            for value_tuple in chunk:

                if parameter.variable_name_1 == "Machine_dia":
                    mcad.set_value_in_mcad(variable_name="Stator_Lam_Dia", value=value_tuple[0])
                    mcad.set_value_in_mcad(variable_name="AFM_D_Rotor", value=value_tuple[0])
                else:
                    mcad.set_value_in_mcad(parameter.variable_name_1, value_tuple[0])
                if parameter.variable_name_2 != "default":
                    mcad.set_value_in_mcad(parameter.variable_name_2, value_tuple[1])
                if parameter.variable_name_3 != "default":
                    mcad.set_value_in_mcad(parameter.variable_name_3, value_tuple[2])

                if parameter.variable_name_1 == "TorquePointsPerCycle":
                    mcad.set_value_in_mcad(variable_name="AirgapMeshPoints_layers", value=12 * int(value_tuple[0]))
                    mcad.set_value_in_mcad(variable_name="AirgapMeshPoints_mesh", value=12 * int(value_tuple[0]))
                if parameter.variable_name_2 == "TorquePointsPerCycle":
                    mcad.set_value_in_mcad(variable_name="AirgapMeshPoints_layers", value=12 * int(value_tuple[1]))
                    mcad.set_value_in_mcad(variable_name="AirgapMeshPoints_mesh", value=12 * int(value_tuple[1]))
                if parameter.variable_name_3 == "TorquePointsPerCycle":
                    mcad.set_value_in_mcad(variable_name="AirgapMeshPoints_layers", value=12 * int(value_tuple[2]))
                    mcad.set_value_in_mcad(variable_name="AirgapMeshPoints_mesh", value=12 * int(value_tuple[2]))

                MCADDataGenerator.execute_mcad_simulation(
                    mcad_processor=mcad,
                    datamanager=datamanager,
                    input_parameters=input_parameter,
                    output_parameters=output_parameters,
                    path_results=temp_csv_path
                )
        print(f"--------- Finished chunk with {len(chunk)} simulations. Closing MotorCAD-Instanz and deleting temp files------------")


    def parameter_variation(
            self,
            parameter: PARAMETERDICT,
            output_parameters: list[str],
            path_folder_results: Path,
            base_filename: str = "base_first_data.mot"
    ) -> None:

        single_parameter_variation = True

        parameter.value_list_1 = np.arange(parameter.variable_config_1[0], parameter.variable_config_1[1],
                                           (parameter.variable_config_1[1] - parameter.variable_config_1[0]) /
                                           parameter.variable_config_1[2]).tolist()


        if parameter.variable_name_2 != "default" and parameter.variable_name_3 != "default":
            single_parameter_variation = False
            parameter.value_list_2 = np.arange(parameter.variable_config_2[0], parameter.variable_config_2[1],
                                               (parameter.variable_config_2[1] - parameter.variable_config_2[0]) /
                                               parameter.variable_config_2[2]).tolist()
            parameter.value_list_3 = np.arange(parameter.variable_config_3[0], parameter.variable_config_3[1],
                                               (parameter.variable_config_3[1] - parameter.variable_config_3[0]) /
                                               parameter.variable_config_3[2]).tolist()
            value_list = list(product(parameter.value_list_1, parameter.value_list_2, parameter.value_list_3))
        elif parameter.variable_name_2 != "default" and parameter.variable_name_3 == "default":
            single_parameter_variation = False
            parameter.value_list_2 = np.arange(parameter.variable_config_2[0], parameter.variable_config_2[1],
                                               (parameter.variable_config_2[1] - parameter.variable_config_2[0]) /
                                               parameter.variable_config_2[2]).tolist()
            value_list = list(product(parameter.value_list_1, parameter.value_list_2))
        else:
            value_list = list(zip(parameter.value_list_1))

        max_progress_steps = len(value_list)
        num_of_workers = min(self.processmanager.set_number_of_workers(number_of_kernels=config_process.num_kernel), max_progress_steps)
        value_chunks = self.datamanager.split_chunks(values=value_list, number_of_chunks=num_of_workers)

        path_temp_csv_folder = Path(config_paths.PATH_DATA_MCAD_RESULTS / "temp_csv")
        path_temp_csv_folder.mkdir(exist_ok=True)

        print(f"--------- Executing {max_progress_steps} simulations. Some might be dropped ------------")

        with ProcessPoolExecutor(max_workers=num_of_workers) as executor:
            futures = [
                executor.submit(
                    MCADDataGenerator.func,
                    value_chunk,
                    parameter,
                    self.parameter_list,
                    output_parameters,
                    path_temp_csv_folder,
                    base_filename,
                    self.datamanager
                ) for value_chunk in value_chunks
            ]
            for future in futures:
                future.result()


        csv_path = self.create_csv_path(path_folder_results, parameter, single_parameter_variation)
        self.datamanager.create_csv_file(
            path=csv_path,
            columns_list=self.parameter_list + output_parameters
        )
        self.datamanager.write_from_csv_in_csv(
            input_folder=path_temp_csv_folder,
            output_file=csv_path,
        )
        self.datamanager.delete_folder(path_temp_csv_folder)
        self.datamanager.delete_folder(config_paths.PATH_MCAD_DATA / config_paths.PATH_TEMP_FILES)


class BasicDataGenerator:

    def __init__(self):
     pass
    def __enter__(self):
     return self
    def __exit__(self, exc_type, exc_val, exc_tb):
     return False

    @staticmethod
    def sin_exp_function(t: float) -> float:
        y = 100 * np.sin(0.8*t) * np.exp(-t/50)
        return y
    @staticmethod
    def sin_function(t: float) -> float:
        y = 100 * np.sin(0.8*t)
        return y
    @staticmethod
    def exp_function(t: float) -> float:
        y = 100 * np.exp(-t/50)
        return y
    @staticmethod
    def linear_function(t: float) -> float:
        y = 1/50 * t + 10
        return y
    @staticmethod
    def quadratic_function(t: float) -> float:
        y = 1/50 * t**2 + 10
        return y


    def data_points(
            self,
            func: str,
            time_start: float = 0.0,
            time_end: float = 100.0,
            number_of_data_points: float = 100
    ) -> pd.DataFrame:
        time = np.arange(time_start, time_end, (time_end-time_start) / number_of_data_points)
        y: list[float] = []

        match func:
            case "sin":
                y = [self.sin_function(t) for t in time]
            case "sin_exp":
                y = [self.sin_exp_function(t) for t in time]
            case "exp":
                y = [self.exp_function(t) for t in time]
            case "linear":
                y = [self.linear_function(t) for t in time]
            case "quadratic":
                y = [self.quadratic_function(t) for t in time]
            case _:
                raise ValueError(f"Function '{func}' is not supported. Please choose 'sin' or 'exp'.")


        data = pd.DataFrame(
            list(zip(time, y)),
            columns=['time', 'y']
        )

        return data