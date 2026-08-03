import math
from itertools import product
from typing import Generator

from modules.utilitis import mymath
from src.config.config_parameter_lists import *


def _range_tuple_to_list(parameter: tuple | list, steps) -> list:

    if isinstance(parameter, tuple):
        step_size = math.floor(((parameter[1] - parameter[0]) / steps) / 10) * 10
        parameter_list = []
        for i in range(int(steps)):
            parameter_list.append(parameter[0] + i * step_size)
        return parameter_list
    else:
        return parameter



def _calc_shaft_dia(rotor_dia: float, factor: float) -> tuple[float, float]:
    shaft_dia = rotor_dia * factor
    difference = rotor_dia - shaft_dia
    return shaft_dia, difference
def _calc_magnet_thickness(difference_rotor_shaft_dia: float, factor: float) -> tuple[float, float]:
    half_diff_rotor_shaft_dia = difference_rotor_shaft_dia / 2
    magnet_thickness = half_diff_rotor_shaft_dia * factor
    return magnet_thickness, half_diff_rotor_shaft_dia
def _calc_magnet_embed_depth(half_diff_rotor_shaft_dia: float, magnet_thickness: float, factor: float) -> float:
    magnet_embed_depth = (half_diff_rotor_shaft_dia - magnet_thickness) * factor
    return magnet_embed_depth
def _calc_slot_width(rotor_dia: float, magnet_thickness: float, magnet_embed_depth: float, slot_number: int, factor_slot_width) -> float:
    dia_mag_i = (rotor_dia - 2 * magnet_embed_depth - 2 * magnet_thickness)
    dia_mag_a = (rotor_dia - 2 * magnet_embed_depth)
    mean_dia_of_mag_length = (dia_mag_a + dia_mag_i) / 2
    max_slot_width = math.pi * mean_dia_of_mag_length / slot_number
    slot_width = round(max_slot_width * factor_slot_width, 3)
    return slot_width
def _calc_slot_opening(slot_width: float, factor_slot_opening: float) -> float:
    slot_opening = round(slot_width * factor_slot_opening, 3)
    return slot_opening


def _calc_radial_parameters(
        rotor_dia: float,
        slot_number: int,
        factor_shaft_dia: float,
        factor_magnet_thickness: float,
        factor_magnet_embed_depth: float,
        factor_slot_width: float,
        factor_slot_opening: float,
) -> dict[str, float]:

    shaft_dia, difference_rotor_shaft_dia = _calc_shaft_dia(
        rotor_dia,
        factor_shaft_dia
    )
    magnet_thickness, half_diff_rotor_shaft_dia = _calc_magnet_thickness(
        difference_rotor_shaft_dia,
        factor_magnet_thickness
    )
    magnet_embed_depth = _calc_magnet_embed_depth(
        half_diff_rotor_shaft_dia,
        magnet_thickness,
        factor_magnet_embed_depth
    )
    slot_width = _calc_slot_width(
        rotor_dia,
        magnet_thickness,
        magnet_embed_depth,
        slot_number,
        factor_slot_width
    )
    slot_opening = _calc_slot_opening(slot_width, factor_slot_opening)
    stator_dia = rotor_dia


    radial_parameters = {
        "Stator_Lam_Dia": stator_dia,
        "Stator_Bore": shaft_dia + 0.01,
        "AFM_D_Rotor": rotor_dia,
        "Shaft_Dia": shaft_dia,
        "Shaft_Hole_Diameter": 0.0,
        "Magnet_Thickness": magnet_thickness,
        "Magnet_Embed_Depth": magnet_embed_depth,
        "Slot_Width": slot_width,
        "Slot_Opening": slot_opening,
    }

    return radial_parameters


def do_radial_variation(
        number_of_datapoints: int = 1000,
        slot_number: int = 18,
) -> Generator[dict[str, float], None, None]:

    a, b, c, d, e, f = mymath.best_factors(number_of_datapoints, k=6)

    rotor_diameters = _range_tuple_to_list(ROTOR_DIAMETER, a)
    factors_shaft_dia = _range_tuple_to_list(FACTORS_SHAFT_DIA, b)
    factors_magnet_thickness = _range_tuple_to_list(FACTORS_MAGNET_TICKNESS, c)
    factors_magnet_embed_depth = _range_tuple_to_list(FACTORS_EMBED_DEPTH, d)
    factor_slot_width = _range_tuple_to_list(FACTORS_SLOT_WIDTH, e)
    factor_slot_opening = _range_tuple_to_list(FACTORS_SLOT_OPENING, f)


    for combination in product(
            rotor_diameters,
            factors_shaft_dia,
            factors_magnet_thickness,
            factors_magnet_embed_depth,
            factor_slot_width,
            factor_slot_opening
    ):
        yield _calc_radial_parameters(
            combination[0],
            slot_number,
            combination[1],
            combination[2],
            combination[3],
            combination[4],
            combination[5]
        )

def do_axial_variation(
        number_of_datapoints: int = 1000,
) -> Generator[dict[str, float], None, None]:

    a, b, c, d, e = mymath.best_factors(number_of_datapoints, k=5)

    stator_lengths = _range_tuple_to_list(STATOR_LAM_LENGTH, a)
    rotor_lengths = _range_tuple_to_list(ROTOR_LAM_LENGTH, b)
    magnet_lengths = _range_tuple_to_list(MAGNET_LENGTH, c)
    airgaps = _range_tuple_to_list(AIRGAP, d)
    slot_depth = _range_tuple_to_list(FACTORES_SLOT_DEPTH, e)


    for combination in product(stator_lengths, rotor_lengths, magnet_lengths, airgaps, slot_depth):
        yield {
            "Motor_Length": 2*combination[0] + combination[1] + 2*combination[2] + 2*combination[3] + 2,
            "Stator_Lam_Length": combination[0],
            "Rotor_Lam_Length": combination[1],
            "Magnet_Length": combination[2],
            "Airgap": combination[3],
            "Slot_Depth": combination[0] * combination[4]
        }

def do_operation_point_variation(
        number_of_datapoints: int = 1000,
) -> Generator[dict[str, float], None, None]:

    a, b, c = mymath.best_factors(number_of_datapoints, k=3)

    peak_current = _range_tuple_to_list(PEAK_CURRENT, a)
    shaft_speed = _range_tuple_to_list(SHAFT_SPEED, b)
    phase_advance = _range_tuple_to_list(PHASE_ADVANCE, c)

    for combination in product(peak_current, shaft_speed, phase_advance):
        yield {
            "PeakCurrent": combination[0],
            "ShaftSpeed": combination[1],
            "PhaseAdvance": combination[2]
        }

def do_winding_variation(number_of_datapoints: int = 1000) -> Generator[dict[str, float], None, None]:
    a, b = mymath.best_factors(number_of_datapoints, k=2)
    print(a, b)
    for x, y in mymath.valid_pairs(x_max=a, y_max=b):
        yield {"Slot_Number": x, "Pole_Number": y}