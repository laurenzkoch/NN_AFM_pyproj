import csv
from dataclasses import dataclass
from typing import Callable
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import math
from pathlib import Path

from src.config import config_paths
from src.modules import process_managing, data_managing
import ansys.motorcad.core as pymotorcad

@dataclass
class PARAMETERDICT:
    variable_name: str
    value_list: list[float]

class MCADDataGenerator(data_managing.DataManager):

    PARAMETER_LIST = [
        "Stator_Lam_Dia",
        "Stator_Bore",
        "Slot_Width",
        "AFM_D_Rotor",
        "Shaft_Dia",
        "Shaft_Hole_Diameter",
        "Magnet_Embed_Depth",
        "Magnet_Thickness",
        "Motor_Length",
        "Stator_Lam_Length",
        "Slot_Depth",
        "Rotor_Lam_Length",
        "Magnet_Length",
        "Slot_Number",
        "Pole_Number",
        "Airgap",
        "MagPhases",
        "PeakCurrent",
        "PhaseAdvance",
    ]


    def __init__(self, start_eduvpn: bool = True):
        super().__init__()
        self.processmanager = process_managing.ProcessManager()
        self.eduVPN_process = None

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
       file_path = config_paths.PATH_MCAD_DATA / filename
       if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

       self.mCAD_process.load_from_file(str(file_path))

    """
    def take_parameter(self, input_df: pd.DataFrame, output_parameters: list[str]) -> None:
        print("---------Preparing input parameter------------")
        self.input_value_array_df, self.number_of_combinations = self.generate_input_value_array_df(input_df)
        self.output_parameters = output_parameters"""


    """ 
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

        return input_parameter_df, number_of_combinations"""


    def test_boundary_conditions(self, values: list[float]) -> bool:

        keys = self.PARAMETER_LIST

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


    def get_values_from_mcad(self, parameter_list: list[str]) -> list[float]:

        value_row = []
        for variable in parameter_list:
            try:
                value_row.append(
                    float(self.mCAD_process.get_variable(variable))
                )
            except Exception as exc:
                raise ValueError(f"Failed to get variable {variable} from MCAD simulation.") from exc

        return value_row


    def do_calculation(self) -> bool:
        try:
            self.mCAD_process.do_magnetic_calculation()
            return True
        except Exception as exc:
            print("Warning: Motor-CAD rejected this design point.")
            print(f"Motor-CAD error: {exc}")
            return False


    def execute_mcad_simulation(
            self,
            output_parameters: list[str],
            path_results: Path,
            comb_counter: int = 0,
            drop_counter: int = 0
    ) -> tuple[int, int]:

        value_row = self.get_values_from_mcad(self.PARAMETER_LIST)

        if self.test_boundary_conditions(value_row):
            if self.do_calculation():
                value_row.extend(self.get_values_from_mcad(output_parameters))
                self.write_row_in_csv(path_results, value_row)
                comb_counter += 1
            else:
                drop_counter += 1
        else:
            drop_counter += 1

        return comb_counter, drop_counter


    @staticmethod
    def create_csv_file(parameter_name: str, columns_list: list[str]):
        print(f"---------Creating CSV file for {parameter_name}-variation------------")

        current_time = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%d.%m.%Y_%H-%M-%S_%Z")

        filename = f"Results_EMagnetic_simulation_{parameter_name}_{current_time}.csv"

        config_paths.PATH_DATA_MCAD_RESULTS.mkdir(exist_ok=True)
        path_results = config_paths.PATH_DATA_MCAD_RESULTS / filename

        with open(path_results, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";", quotechar='"')
            writer.writerow(columns_list)

        return path_results


    def single_parameter_variation(self, parameter: PARAMETERDICT, output_parameters: list[str], base_filename: str = "base_first_data.mot") -> None:

        path_results = self.create_csv_file(
            parameter_name=parameter.variable_name,
            columns_list=self.PARAMETER_LIST + output_parameters
        )
        self.open_motorcad_file(filename=base_filename)

        drop_counter = 0
        comb_counter = 0
        max_progress_steps = len(parameter.value_list)
        tracking_progress = self.track_progress(max_progress_steps)

        for value in parameter.value_list:

            self.set_values_in_mcad(parameter.variable_name, value)
            comb_counter, drop_counter = self.execute_mcad_simulation(
                output_parameters=output_parameters,
                path_results=path_results,
                comb_counter=comb_counter,
                drop_counter=drop_counter
            )
            tracking_progress(
                comb_counter + drop_counter
            )

        print(f"{drop_counter} points dropped. {comb_counter} points saved.")


    """def multi_parameter_variation(self) -> None:

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
                output_parameters=self.output_parameters,
                path_results=path_results,
                comb_counter=comb_counter,
                drop_counter=drop_counter
            )

            tracking_progress(
                drop_counter + comb_counter
            )


        print(f"{drop_counter} points dropped. {comb_counter} points saved.")"""



