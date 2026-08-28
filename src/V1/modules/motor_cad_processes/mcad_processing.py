from __future__ import annotations
import math
import time
from pathlib import Path

import ansys.motorcad.core as pymotorcad
from V1.config.config_mcad_parameters import OUTPUT_NEURON_LIST, INPUT_NEURON_LIST
from V1.config.config_paths import PATH_MCAD_BASEFILE


class MCADProcessor:

    def __init__(self):
        super().__init__()
        print(f"-----------Starting MCAD: Connection to licence server should have been initialized----------")
        self.mcad_process = pymotorcad.MotorCAD()
        self.mcad_process.set_visible(True)

    def __enter__(self) -> 'MCADProcessor':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            if self.mcad_process is not None:
                try:
                    self.mcad_process.quit()
                    time.sleep(2)
                finally:
                    self.mcad_process = None
        except Exception as exc:
            print(f"Error while closing MCAD: {exc}")
        return None


    def open_motorcad_file(self, file_path: str | Path = PATH_MCAD_BASEFILE) -> None:

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.mcad_process.load_from_file(str(file_path))


    def set_value_in_mcad(self, variable_name: str, value: float) -> None:
        try:
            self.mcad_process.set_variable(variable_name, value)
        except Exception as exc:
            raise ValueError(f"Failed to set variable {variable_name} to value {value}.") from exc


    def get_value_from_mcad(self, variable_name: str) -> float:
        try:
            return float(self.mcad_process.get_variable(variable_name))
        except Exception as exc:
            raise ValueError(f"Failed to get variable {variable_name} from MotorCAD.") from exc


    def do_calculation(self) -> bool:
        try:
            self.mcad_process.do_magnetic_calculation()
            return True
        except Exception as exc:
            print("Warning: Motor-CAD rejected this design point.")
            print(f"Motor-CAD error: {exc}")
            return False


    def execute_mcad_simulation(self) -> tuple[list[float], bool]:

        def get_values(var_list: list[str]) -> list[float]:
            return [self.get_value_from_mcad(var) for var in var_list]

        value_row = get_values(INPUT_NEURON_LIST)

        if self.test_boundary_conditions(INPUT_NEURON_LIST, value_row):
            if self.do_calculation():
                value_row.extend(
                    get_values(OUTPUT_NEURON_LIST)
                )
                return value_row, True
            else:
                for _var in INPUT_NEURON_LIST: print(f"{_var}: {value_row[INPUT_NEURON_LIST.index(_var)]}")
                return value_row, False
        else:
            return value_row, False


    @staticmethod
    def test_boundary_conditions(parameter_list: list[str], values: list[float]) -> bool:

        keys = parameter_list

        if len(keys) != len(values):
            raise ValueError("Anzahl der keys stimmt nicht mit Anzahl der values überein")

        parameter_dict = {
            k: float(v)
            for k, v in zip(keys, values)
        }

        # radial - stator
        stator_lam_dia = parameter_dict["Stator_Lam_Dia"]
        stator_bore = parameter_dict["Stator_Bore"]
        slot_width = parameter_dict["Slot_Width"]

        # radial - rotor
        afm_d_rotor = parameter_dict["AFM_D_Rotor"]
        shaft_dia = parameter_dict["Shaft_Dia"]
        shaft_hole_diameter = parameter_dict["Shaft_Hole_Diameter"]
        magnet_embed_depth = parameter_dict["Magnet_Embed_Depth"]
        magnet_thickness = parameter_dict["Magnet_Thickness"]

        # axial
        motor_length = parameter_dict["Motor_Length"]
        stator_lam_length = parameter_dict["Stator_Lam_Length"]
        slot_depth = parameter_dict["Slot_Depth"]
        rotor_lam_length = parameter_dict["Rotor_Lam_Length"]
        magnet_length = parameter_dict["Magnet_Length"]

        # winding
        slot_opening = parameter_dict["Slot_Opening"]
        tooth_tip_depth = parameter_dict["Tooth_Tip_Depth"]
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
        winding_ratio = slot_number / magphases / winding_gcd

        # radial - stator
        if stator_bore >= stator_lam_dia:
            print("Warning: Stator bore is greater than stator lamination diameter.")
            return False
        if stator_bore <= shaft_dia:
            print("Warning: Stator bore is less than shaft diameter.")
            return False
        # radial - rotor
        if dia_mag_i <= shaft_dia:
            print("Warning: Magnets too thick for rotor diameter or rotor diameter is less than shaft diameter.")
            return False
        if shaft_hole_diameter >= stator_lam_dia:
            print("Warning: Shaft hole diameter is greater than stator lamination diameter.")
            return False


        # axial
        if motor_length <= (2 * (stator_lam_length + airgap + magnet_length) + rotor_lam_length):
            print(f"Warning: Motor length is too short.")
            return False
        if slot_depth >= stator_lam_length:
            print("Warning: Slot depth is greater than stator lamination length.")
            return False


        # winding
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
        if tooth_tip_depth >= slot_depth:
            print("Warning: Tooth tip depth is greater than stator tip depth.")
            return False
        if slot_opening > slot_width:
            print("Warning: Slot opening is greater than tooth tip width.")
            return False


        # magnet length
        if activ_length <= 0:
            print("Warning: Magnet length is less than 0")
            return False

        return True