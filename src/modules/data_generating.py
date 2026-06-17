import csv
from typing import Callable
import pandas as pd
from itertools import product
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import math
from pathlib import Path

from src.config import config_paths
from src.modules import process_managing
import ansys.motorcad.core as pymotorcad

class MCADDataGenerator:

    PATH_MOTORCAD_DATA = config_paths.PATH_MCAD_DATA
    PATH_RESULTS = config_paths.PATH_DATA_MCAD_RESULTS
    PATH_RESULTS.mkdir(exist_ok=True)

    def __init__(self, start_eduvpn: bool = True):
        self.processmanager = process_managing.ProcessManager()
        self.eduVPN_process = None
        self.input_value_array_df = pd.DataFrame(
            columns=["name", "value_array"],
        )
        self.output_parameters: list[str] = []
        self.number_of_combinations = 0

        if start_eduvpn:
            print("-----------Starting eduVPN----------")
            self.eduVPN_process = self.processmanager.start_process(config_paths.PATH_eduVPN_EXE)

        print("-----------Starting MCAD----------")
        self.mCAD_process = pymotorcad.MotorCAD()
        self.mCAD_process.set_visible(False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.mCAD_process is not None:
                try:
                    self.mCAD_process.quit()
                finally:
                    self.mCAD_process = None
        finally:
            if self.eduVPN_process is not None:
                self.processmanager.stop_process(self.eduVPN_process)
                self.eduVPN_process  = None


    def open_motorcad_file(self, filename: str = "base_first_data.mot") -> None:
       file_path = self.PATH_MOTORCAD_DATA / filename
       if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

       self.mCAD_process.load_from_file(str(file_path))


    def take_parameter(self, input_df: pd.DataFrame, output_parameters: list[str]) -> None:
        print("---------Preparing input parameter------------")
        self.input_value_array_df, self.number_of_combinations = self.generate_input_value_array_df(input_df)
        self.output_parameters = output_parameters


    @staticmethod
    def generate_input_value_array_df(input_parameter_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:

        input_parameter_df["value_array"] = [
            np.arange(
                float(row["min_value"]),
                float(row["max_value"]),
                float(row["step_size"])
            ).tolist()
            for _, row in input_parameter_df.iterrows()
        ]

        input_parameter_df.drop(columns=["min_value", "max_value", "step_size"], inplace=True)

        number_of_combinations = 1
        for i in range(len(input_parameter_df["value_array"]) - 1):
            number_of_combinations = input_parameter_df["value_array"].iloc[i].__len__() * number_of_combinations

        print(f"If Multi parameter variation: {number_of_combinations} combinations will be created.")

        return input_parameter_df, number_of_combinations


    def test_boundary_conditions(self) -> bool:

        mcad = self.mCAD_process

        slot_depth = float(mcad.get_variable("Slot_Depth"))
        motor_length = float(mcad.get_variable("Motor_Length"))
        magnet_length = float(mcad.get_variable("Magnet_Length"))
        stator_lam_length = float(mcad.get_variable("Stator_Lam_Length"))
        stator_bore = float(mcad.get_variable("Stator_Bore"))
        afm_d_rotor = float(mcad.get_variable("AFM_D_Rotor"))
        shaft_dia = float(mcad.get_variable("Shaft_Dia"))
        magnet_thickness = float(mcad.get_variable("Magnet_Thickness"))
        slot_number = int(mcad.get_variable("Slot_Number"))
        pole_number = int(mcad.get_variable("Pole_Number"))
        airgap = float(mcad.get_variable("Airgap"))
        stator_lam_dia = float(mcad.get_variable("Stator_Lam_Dia"))
        magphases = int(mcad.get_variable("MagPhases"))
        mcf_pole_slot = math.gcd(slot_number, pole_number)
        _, winding_mod = divmod(slot_number, (magphases * mcf_pole_slot))

        if slot_depth >= stator_lam_length:
            print("Warning: Slot depth is greater than stator lamination length. This is not allowed.")
            return False
        if motor_length <= stator_lam_length:
            print("Warning: Motor length is less than stator lamination length. This is not allowed.")
            return False
        if magnet_length <= airgap:
            print("Warning: Magnet length is less than airgap. This is not allowed.")
            return False
        if magnet_length >= stator_lam_length:
            print("Warning: Magnet length is greater than stator lamination length. This is not allowed.")
            return False
        if stator_bore >= stator_lam_dia:
            print("Warning: Stator bore is greater than stator lamination diameter. This is not allowed.")
            return False
        if stator_bore <= shaft_dia:
            print("Warning: Stator bore is less than shaft diameter. This is not allowed.")
            return False
        if afm_d_rotor <= stator_bore:
            print("Warning: AFM diameter of rotor is less than stator bore. This is not allowed.")
            return False
        if (afm_d_rotor - magnet_thickness)  >= stator_lam_dia:
            print("Warning: AFM diameter of rotor minus magnet thickness is greater than stator lamination diameter. This is not allowed.")
            return False
        if ((afm_d_rotor - shaft_dia ) * 0.5)  <= magnet_thickness:
            print("Warning: AFM diameter of rotor minus shaft diameter is less than or equal to magnet thickness. This is not allowed.")
            return False
        if  winding_mod != 0:
             print("Warning: Slot number is not divisible by magphases*mcf pole slot. This is not allowed.")
             return False

        return True


    @staticmethod
    def track_progress(number_of_total_progress_steps: int) -> Callable[..., float]:
        starttime = time.time()

        def inner_track_progress(progress_step: int) -> float:
            progress_percent_new = progress_step / number_of_total_progress_steps * 100

            elapsedtime = time.time() - starttime
            remainingtime = (elapsedtime / progress_step * (number_of_total_progress_steps - progress_step))
            days, remainder = divmod(remainingtime, 24 * 3600)
            hours, remainder = divmod(remainder, 3600)
            minutes, seconds = divmod(remainder, 60)

            print(f"{progress_percent_new:.2f}% done. Remaining time: {int(days)}d {int(hours)}h {int(minutes)}min {seconds:.2f}s")
            return progress_percent_new

        return inner_track_progress

    def set_values_in_mcad(self, variable_name: str, value: float) -> None:
        try:
            self.mCAD_process.set_variable(variable_name, value)
        except Exception as exc:
            raise ValueError(f"Failed to set variable {variable_name} to value {value}.") from exc


    def get_values_from_mcad(self) -> list[float]:

        def make_parameter_list() -> list[str]:
            parameter_list: list[str] = self.input_value_array_df["name"].tolist()
            parameter_list.extend(self.output_parameters)
            return parameter_list
        parameters_list = make_parameter_list()

        value_row = []
        for variable in parameters_list:
            try:
                value_row.append(float(self.mCAD_process.get_variable(variable)))
            except Exception as exc:
                raise ValueError(f"Failed to get variable {variable} from MCAD simulation.") from exc

        return value_row

    @staticmethod
    def write_row_in_csv(path_results: Path, value_row: list[float]) -> None:

        with open(path_results, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";", quotechar='"')
            formatted_value_row = [
                f"{x:.4f}".replace(".", ",") if isinstance(x, float) else x
                for x in value_row
            ]
            writer.writerow(formatted_value_row)


    def do_calculation(self) -> None:
        try:
            self.mCAD_process.do_magnetic_calculation()
        except Exception as exc:
            raise ValueError(f"Failed to execute MCAD calculation.") from exc


    def execute_mcad_simulation(
            self,
            path_results: Path,
            comb_counter: int = 0,
            drop_counter: int = 0
    ) -> tuple[int, int]:

        print("---------Starting MCAD simulation------------")

        if self.test_boundary_conditions():
            self.do_calculation()
            value_row = self.get_values_from_mcad()
            self.write_row_in_csv(path_results, value_row)
            comb_counter += 1
        else:
            drop_counter += 1

        return comb_counter, drop_counter

    def create_csv_file(self):
        print("---------Creating CSV file------------")

        current_time = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%d.%m.%Y_%H-%M-%S_%Z")

        filename = f"Results_EMagnetic_simulation_{current_time}.csv"
        path_results = self.PATH_RESULTS / filename

        with open(path_results, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";", quotechar='"')
            writer.writerow(self.input_value_array_df["name"].tolist() + self.output_parameters)

        return path_results



    def single_parameter_variation(self) -> None:

        path_results = self.create_csv_file()

        drop_counter = 0
        comb_counter = 0
        max_progress_steps = sum(len(self.input_value_array_df["value_array"].iloc[i]) for i in range(len(self.input_value_array_df)))
        tracking_progress = self.track_progress(max_progress_steps)

        def get_value_array(var_name: str) -> pd.Series:
            index = self.input_value_array_df["name"].tolist().index(var_name)
            return self.input_value_array_df["value_array"].iloc[index]

        for variable_name in self.input_value_array_df["name"]:
            self.open_motorcad_file()
            print(f"---------Starting parameter variation: {variable_name}------------")

            for value in get_value_array(var_name=variable_name):

                self.set_values_in_mcad(variable_name, value)
                comb_counter, drop_counter = self.execute_mcad_simulation(
                    path_results=path_results,
                    comb_counter=comb_counter,
                    drop_counter=drop_counter
                )

                tracking_progress(
                    comb_counter + drop_counter
                )

        print(f"{drop_counter} points dropped. {comb_counter} points saved.")


    def multi_parameter_variation(self) -> None:

        path_results = self.create_csv_file()

        drop_counter = 0
        comb_counter = 0
        tracking_progress = self.track_progress(self.number_of_combinations)

        def get_combination_value(var_name: str, combination_tuple: tuple) -> float:
            variable_index = self.input_value_array_df["name"].tolist().index(var_name)
            return combination_tuple[variable_index]


        for combination in product(*self.input_value_array_df["value_array"].tolist()):

            if comb_counter > 1000000:
                raise OverflowError("Warning: The number of value combinations is too large."
                                    " Reducing the step sizes, intervall size or number of parameters.")

            for variable in self.input_value_array_df["name"]:
                value = get_combination_value(var_name=variable, combination_tuple=combination)
                self.set_values_in_mcad(variable_name=variable, value=value)

            comb_counter, drop_counter = self.execute_mcad_simulation(
                path_results=path_results,
                comb_counter=comb_counter,
                drop_counter=drop_counter
            )

            tracking_progress(
                drop_counter + comb_counter
            )


        print(f"{drop_counter} points dropped. {comb_counter} points saved.")



