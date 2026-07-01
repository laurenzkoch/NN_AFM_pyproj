from dataclasses import dataclass, field
from itertools import product
from typing import Callable
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from src.config import config_paths, config_parameter_lists, config_process
from src.modules import process_managing, data_managing
import ansys.motorcad.core as pymotorcad

@dataclass
class PARAMETERDICT:
    variable_name_1: str = ""
    variable_name_2: str = "default"
    variable_name_3: str = "default"
    value_list_1: list[float] = field(default_factory=list)
    value_list_2: list[float] = field(default_factory=list)
    value_list_3: list[float] = field(default_factory=list)

class MCADDataGenerator:

    parameter_list = config_parameter_lists.PARAMETER_LIST

    def __init__(self, start_eduvpn: bool = True, init_mcad: bool = False):
        self.processmanager = process_managing.ProcessManager()
        self.DataManager = data_managing.DataManager()
        self.eduVPN_process = None
        self.mcad_process = None

        if start_eduvpn:
            print("-----------Starting eduVPN----------")
            self.eduVPN_process = self.processmanager.start_process(config_paths.PATH_eduVPN_EXE)
        if init_mcad:
            print(f"-----------Starting MCAD----------")
            self.mcad_process = pymotorcad.MotorCAD()
            self.mcad_process.set_visible(False)


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.mcad_process is not None:
                try:
                    self.mcad_process.quit()
                finally:
                    self.mcad_process = None
        finally:
            if self.eduVPN_process is not None:
                self.processmanager.stop_process(self.eduVPN_process)
                self.eduVPN_process  = None


    @staticmethod
    def open_motorcad_file(mcad_process: pymotorcad.MotorCAD, filename: str = "base_first_data.mot") -> None:
       file_path = config_paths.PATH_MCAD_DATA / filename
       if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

       mcad_process.load_from_file(str(file_path))


    def test_boundary_conditions(self, values: list[float]) -> bool:

        keys = self.parameter_list

        if len(keys) != len(values):
            raise ValueError("Anzahl der keys stimmt nicht mit Anzahl der values überein")

        parameter_dict = {
            k: float(v)
            for k, v in zip(keys, values)
        }

        #radial - stator
        stator_lam_dia = parameter_dict["Stator_Lam_Dia"]
        stator_bore = parameter_dict["Stator_Bore"]
        slot_width = parameter_dict["Slot_Width"]

        #radial - rotor
        afm_d_rotor = parameter_dict["AFM_D_Rotor"]
        shaft_dia = parameter_dict["Shaft_Dia"]
        shaft_hole_diameter = parameter_dict["Shaft_Hole_Diameter"]
        magnet_embed_depth = parameter_dict["Magnet_Embed_Depth"]
        magnet_thickness = parameter_dict["Magnet_Thickness"]

        #axial
        motor_length = parameter_dict["Motor_Length"]
        stator_lam_length = parameter_dict["Stator_Lam_Length"]
        slot_depth = parameter_dict["Slot_Depth"]
        rotor_lam_length = parameter_dict["Rotor_Lam_Length"]
        magnet_length = parameter_dict["Magnet_Length"]

        #winding
        slot_number = int(parameter_dict["Slot_Number"])
        pole_number = int(parameter_dict["Pole_Number"])
        airgap = parameter_dict["Airgap"]
        magphases = int(parameter_dict["MagPhases"])

        dia_mag_i = (afm_d_rotor - 2 * magnet_embed_depth - 2 * magnet_thickness)
        dia_mag_a = (afm_d_rotor - 2 * magnet_embed_depth)
        activ_length = min(stator_lam_dia, dia_mag_a) - max(stator_bore, dia_mag_i)
        mean_dia_of_mag_length = (dia_mag_a + dia_mag_i) / 2
        stator_width_per_slot = math.pi * mean_dia_of_mag_length / slot_number
        winding_gcd = math.gcd(slot_number, pole_number)
        winding_denominator = magphases * winding_gcd
        winding_ratio = slot_number / winding_denominator


        #radial - stator
        if stator_bore >= stator_lam_dia:
            print("Warning: Stator bore is greater than stator lamination diameter.")
            return False
        if stator_bore <= shaft_dia:
            print("Warning: Stator bore is less than shaft diameter.")
            return False
        #radial - rotor
        if dia_mag_i  <= shaft_dia:
            print("Warning: Magnets too thick for rotor diameter or rotor diameter is less than shaft diameter.")
            return False
        if shaft_hole_diameter >= stator_lam_dia:
            print("Warning: Shaft hole diameter is greater than stator lamination diameter.")
            return False

        #axial
        if motor_length <= (2 * (stator_lam_length + airgap + magnet_length) + rotor_lam_length):
            print(f"Warning: Motor length is too short.")
            return False
        if slot_depth >= stator_lam_length:
            print("Warning: Slot depth is greater than stator lamination length.")
            return False

        #winding
        if not winding_ratio.is_integer() or winding_ratio < 1:
            print(
                "Warning: Winding is not feasible. "
                f"Slot_Number / MagPhases / gcd(Pole_Number, Slot_Number) = "
                f"{slot_number} / {magphases} / {winding_gcd} = {winding_ratio:.6f}."
            )
            return False

        if slot_width >= stator_width_per_slot:
            print("Warning: Slot width is greater than stator width per slot.")
            return False


        #magnet length
        if activ_length <= 0:
            print("Warning: Magnet length is less than 0")
            return False

        return True


    @staticmethod
    def track_progress(number_of_total_progress_steps: int) -> Callable[..., float]:
        starttime = time.time()

        def inner_track_progress(progress_step: int) -> float:
            progress_percent_new = progress_step / number_of_total_progress_steps * 100

            elapsedtime = time.time() - starttime
            remaining_steps = number_of_total_progress_steps - progress_step
            approx_time_per_step = elapsedtime / progress_step
            remainingtime = (approx_time_per_step + 2*13) / 3 * remaining_steps #13s per simulation guessed; mean remaining time
            days, remainder = divmod(remainingtime, 24 * 3600)
            hours, remainder = divmod(remainder, 3600)
            minutes, seconds = divmod(remainder, 60)

            print(f"{progress_percent_new:.2f}% done. Remaining time: {int(days)}d {int(hours)}h {int(minutes)}min {seconds:.2f}s")
            return elapsedtime

        return inner_track_progress


    @staticmethod
    def set_value_in_mcad(mcad_process: pymotorcad.MotorCAD, variable_name: str, value: float) -> None:
        try:
            mcad_process.set_variable(variable_name, value)
        except Exception as exc:
            raise ValueError(f"Failed to set variable {variable_name} to value {value}.") from exc


    @staticmethod
    def get_values_from_mcad(mcad_process: pymotorcad.MotorCAD, parameter_list: list[str]) -> list[float]:
        value_row = []
        for variable in parameter_list:
            try:
                value_row.append(
                    float(mcad_process.get_variable(variable))
                )
            except Exception as exc:
                raise ValueError(f"Failed to get variable {variable} from MCAD simulation.") from exc

        return value_row


    @staticmethod
    def do_calculation(mcad_process: pymotorcad.MotorCAD) -> bool:
        try:
            mcad_process.do_magnetic_calculation()
            return True
        except Exception as exc:
            print("Warning: Motor-CAD rejected this design point.")
            print(f"Motor-CAD error: {exc}")
            return False


    def execute_mcad_simulation(
            self,
            mcad_process: pymotorcad.MotorCAD,
            output_parameters: list[str],
            path_results: Path,
    ) -> None:

        value_row = self.get_values_from_mcad(mcad_process, self.parameter_list)

        if self.test_boundary_conditions(value_row):
            if self.do_calculation(mcad_process=mcad_process):
                value_row.extend(self.get_values_from_mcad(mcad_process, output_parameters))
                self.DataManager.write_row_in_csv(path_results, value_row)


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
    def temp_csv_path(path_folder: Path) -> Path:

        current_time = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%d.%m.%Y_%H-%M-%S_%Z.%f")
        filename = f"temp_{current_time}.csv"

        return path_folder / filename


    @staticmethod
    def func(
            chunk: list[tuple[float, float, float]],
            parameter: PARAMETERDICT,
            output_parameters: list[str],
            temp_folder_path: Path,
            base_filename: str = "base_first_data.mot",
    ) -> None:

        with MCADDataGenerator(start_eduvpn=False, init_mcad=True) as generator:

            mcad_process = generator.mcad_process if generator.mcad_process is not None else pymotorcad.MotorCAD()
            base_file = generator.DataManager.create_temp_file(Path(config_paths.PATH_MCAD_DATA / base_filename))

            generator.open_motorcad_file(mcad_process=mcad_process, filename=base_file.name)
            temp_csv_path = generator.temp_csv_path(temp_folder_path)

            for value_tuple in chunk:

                generator.set_value_in_mcad(mcad_process, parameter.variable_name_1, value_tuple[0])
                if parameter.variable_name_2 != "default":
                    generator.set_value_in_mcad(mcad_process, parameter.variable_name_2, value_tuple[1])
                if parameter.variable_name_3 != "default":
                    generator.set_value_in_mcad(mcad_process, parameter.variable_name_3, value_tuple[2])

                generator.execute_mcad_simulation(
                    mcad_process=mcad_process,
                    output_parameters=output_parameters,
                    path_results=temp_csv_path
                )

            generator.DataManager.delete_temp_file_and_related_folder(base_file)    


    def parameter_variation(
            self,
            parameter: PARAMETERDICT,
            output_parameters: list[str],
            path_folder_results: Path,
            base_filename: str = "base_first_data.mot"
    ) -> None:

        single_parameter_variation = True

        if parameter.variable_name_2 != "default" and parameter.variable_name_3 != "default":
            single_parameter_variation = False
            value_list = list(product(parameter.value_list_1, parameter.value_list_2, parameter.value_list_3))
        elif parameter.variable_name_2 != "default" and parameter.variable_name_3 == "default":
            single_parameter_variation = False
            value_list = list(product(parameter.value_list_1, parameter.value_list_2))
        else:
            value_list = list(zip(parameter.value_list_1))

        max_progress_steps = len(value_list)
        num_of_workers = self.processmanager.set_number_of_workers(number_of_kernels=config_process.num_kernel)
        value_chunks = self.DataManager.split_chunks(values=value_list, number_of_chunks=num_of_workers)

        path_temp_folder = Path(config_paths.PATH_DATA_MCAD_RESULTS / "temp_csv")
        path_temp_folder.mkdir(exist_ok=True)

        print(f"--------- Executing {max_progress_steps} simulations. Some might be dropped ------------")

        with ProcessPoolExecutor(max_workers=num_of_workers) as executor:
            futures = [
                executor.submit(
                    MCADDataGenerator.func,
                    value_chunk,
                    parameter,
                    output_parameters,
                    path_temp_folder,
                    base_filename
                ) for value_chunk in value_chunks
            ]
            for future in futures:
                future.result()


        csv_path = self.create_csv_path(path_folder_results, parameter, single_parameter_variation)
        self.DataManager.create_csv_file(
            path=csv_path,
            columns_list=self.parameter_list + output_parameters
        )
        self.DataManager.write_from_csv_in_csv(
            input_folder=path_temp_folder,
            output_file=csv_path,
        )
        self.DataManager.delete_folder(path_temp_folder)