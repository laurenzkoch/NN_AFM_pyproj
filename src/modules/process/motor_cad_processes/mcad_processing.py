from __future__ import annotations

import logging
import time
import math

import ansys.motorcad.core as pymotorcad
from src.config.config_mcad_parameters import *
from src.config.config_paths import *

logger = logging.getLogger(__name__)

def _print_design(parameter_list: list[str], value_list: list[float | int]) -> None:
    for _var, _value in zip(parameter_list, value_list):
        print(f"\t\t{_var}: {_value}")
    print("__________________________________________________")

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
    tooth_tip_angle = parameter_dict["Tooth_Tip_Angle"]

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
    stator_width_per_slot = math.pi * stator_bore / slot_number
    pole_pairs = int(pole_number / 2)
    winding_gcd = math.gcd(slot_number, pole_pairs)
    winding_ratio = slot_number / magphases / winding_gcd

    #operation points
    phaseadvanced = parameter_dict["PhaseAdvance"]


    TOL = 1e-6  # mm

    # radial - stator
    if stator_bore >= stator_lam_dia:
        logger.warning("Warning: Stator bore is greater than stator lamination diameter.")
        return False
    if stator_bore <= shaft_dia:
        logger.warning("Warning: Stator bore is less than shaft diameter.")
        return False

    # radial - rotor
    if dia_mag_i <= shaft_dia:
        logger.warning("Warning: Magnets too thick for rotor diameter or rotor diameter is less than shaft diameter.")
        return False
    if shaft_hole_diameter >= stator_lam_dia:
        logger.warning("Warning: Shaft hole diameter is greater than stator lamination diameter.")
        return False


    # axial
    if motor_length < (2 * (stator_lam_length + airgap + magnet_length) + rotor_lam_length) -TOL:
        logger.warning(f"Warning: Motor length is too short.")
        return False
    if slot_depth >= stator_lam_length:
        logger.warning("Warning: Slot depth is greater than stator lamination length.")
        return False
    if rotor_lam_length < 0.25 or magnet_length < 0.25:
        logger.warning("Axial length of rotor/stator/magnets too short")
        return False


    # winding
    if not winding_ratio.is_integer() or winding_ratio < 1:
        logger.warning(
            "Warning: Winding is not feasible. "
            f"Slot_Number / MagPhases / gcd(Pole_Number, Slot_Number) = "
            f"{slot_number} / {magphases} / {winding_gcd} = {winding_ratio:.6f}."
        )
        return False
    if slot_width >= stator_width_per_slot:
        logger.warning("Warning: Slot width is greater than stator width per slot.")
        return False
    if tooth_tip_depth >= slot_depth:
        logger.warning("Warning: Tooth tip depth is greater than stator tip depth.")
        return False
    if slot_opening > slot_width:
        logger.warning("Warning: Slot opening is greater than slot width.")
        return False


    # magnet length
    if activ_length <= 0:
        logger.warning("Warning: Magnet length is less than 0")
        return False


    # operation points
    if phaseadvanced < 0 or phaseadvanced >= 90:
        logger.warning("Warning: Phase advance is out of bounds. It should be in the range [0, 90).")
        return False

    return True



class MCADProcessor:

    def __init__(self) -> None:
        pymotorcad.set_motorcad_exe(str(PATH_MCAD_EXE))
        self.mcad_process: pymotorcad.MotorCAD = pymotorcad.MotorCAD()

    def start(self) -> None:
        self.mcad_process.set_visible(True)

    def close(self) -> None:
        try:
            self.mcad_process.quit()
            time.sleep(2)
        except Exception as exc:
            logger.error(f"Error while closing MCAD: {exc}")

    def __enter__(self) -> 'MCADProcessor':
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
        return None


    def open_motorcad_file(self, file_path: str | Path = PATH_MCAD_BASEFILE) -> None:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

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
        except pymotorcad.MotorCADError as exc:
            logger.error(
                "Motor-CAD calculation failed: %s",
                exc
            )
            return False


    def execute_mcad_simulation(self) -> tuple[list[float] | None, bool]:

        def get_values(var_list: list[str]) -> list[float]:
            return [self.get_value_from_mcad(var) for var in var_list]

        all_parameters: list[str] = list(DEFAULT_VALUES.keys())
        all_values: list[float] = get_values(all_parameters)


        if test_boundary_conditions(all_parameters, all_values):

            if self.do_calculation():
                value_row = get_values(INPUT_NEURON_LIST + OP_PARAMETERS + OUTPUT_NEURON_LIST)

                #print("___________Calculation ran sucessfully___________")
                #_print_design(all_parameters, all_values)
                #_print_design(OUTPUT_NEURON_LIST, output_values)
                return value_row, True

            else:
                logger.error("Calculation failed")
                _print_design(all_parameters, all_values)
                return None, False

        else:
            logger.error("Failed boundary check:")
            _print_design(all_parameters, all_values)
            return None, False