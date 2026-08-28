import math
from itertools import product
from typing import Generator

from V1.modules.utilitis import mymath
from V1.config.config_mcad_parameters import *


def _range_tuple_to_list(parameter: tuple | list, steps) -> list:

    if isinstance(parameter, tuple):
        step_size = (parameter[1] - parameter[0]) / steps

        if step_size > 10:
            step_size = math.floor(step_size / 10) * 10
        elif step_size > 1:
            step_size = math.floor(step_size / 1) * 1
        else:
            step_size = math.floor(step_size / 0.01) * 0.01

        parameter_list = []
        for i in range(int(steps)):
            parameter_list.append(parameter[0] + i * step_size)
        return parameter_list
    else:
        return parameter


def _calc_magnet_thickness(
        rotor_dia: float,
        num_of_points: int
) -> tuple[list[float], float, float]:
    half_diff_rotor_shaft_dia = (rotor_dia - DEFAULT_VALUES["Stator_Bore"]) / 2
    slot_width_constrain_with_zero_embed_depth = rotor_dia - DEFAULT_VALUES["Slot_Width"] * DEFAULT_VALUES["Slot_Number"] / math.pi
    max_magnet_thickness = min(half_diff_rotor_shaft_dia, slot_width_constrain_with_zero_embed_depth)
    magnet_thickness = (1, max_magnet_thickness)
    magnet_thickness = _range_tuple_to_list(magnet_thickness, num_of_points)
    return magnet_thickness, half_diff_rotor_shaft_dia, slot_width_constrain_with_zero_embed_depth

def _calc_magnet_embed_depth(
        half_diff_rotor_shaft_dia: float,
        slot_width_constrain_with_zero_embed_depth: float,
        magnet_thickness: float,
        num_of_points: int
) -> list[float]:
    max_space_constrain = half_diff_rotor_shaft_dia - magnet_thickness
    slot_width_constrain_embed_depth = 0.5 * (slot_width_constrain_with_zero_embed_depth - magnet_thickness)
    max_magnet_embed_depth = min(max_space_constrain, slot_width_constrain_embed_depth)
    magnet_embed_depth = (0, max_magnet_embed_depth)
    magnet_embed_depth = _range_tuple_to_list(magnet_embed_depth, num_of_points)
    return magnet_embed_depth


def do_radial_variation(
        number_of_datapoints: int = 1000,
) -> Generator[dict[str, float], None, None]:

    a, b, c, = mymath.best_factors(number_of_datapoints, k=3)
    rotor_diameters = _range_tuple_to_list(ROTOR_DIAMETER, a)

    for rotor_dia in rotor_diameters:
        magnet_thicknesses, half_diff_rotor_shaft_dia, slot_width_constrain_with_zero_embed_depth = _calc_magnet_thickness(
            rotor_dia,
            b
        )
        for magnet_thickness in magnet_thicknesses:
            magnet_embed_depths = _calc_magnet_embed_depth(
                half_diff_rotor_shaft_dia,
                slot_width_constrain_with_zero_embed_depth,
                magnet_thickness,
                c
            )
            for magnet_embed_depth in magnet_embed_depths:
                yield {
                    "Stator_Lam_Dia": rotor_dia,
                    "AFM_D_Rotor": rotor_dia,
                    "Magnet_Thickness": magnet_thickness,
                    "Magnet_Embed_Depth": magnet_embed_depth,
                }


def do_axial_variation(
        number_of_datapoints: int = 1000,
) -> Generator[dict[str, float], None, None]:

    a, b, c, d = mymath.best_factors(number_of_datapoints, k=4)

    stator_lengths = _range_tuple_to_list(STATOR_LAM_LENGTH, a)
    rotor_lengths = _range_tuple_to_list(ROTOR_LAM_LENGTH, b)
    magnet_lengths = _range_tuple_to_list(MAGNET_LENGTH, c)
    airgaps = _range_tuple_to_list(AIRGAP, d)

    for combination in product(stator_lengths, rotor_lengths, magnet_lengths, airgaps):
        yield {
            "Motor_Length": 2*combination[0] + combination[1] + 2*combination[2] + 2*combination[3] + 2,
            "Stator_Lam_Length": combination[0],
            "Rotor_Lam_Length": combination[1],
            "Magnet_Length": combination[2],
            "Airgap": combination[3],
        }


def _calc_i_qs(num_of_points: int) -> list[float]:
    i_q_max = PEAK_CURRENT[1] / math.sqrt(2)
    i_q = (0, i_q_max)
    i_qs = _range_tuple_to_list(i_q, num_of_points)
    return i_qs
def  _calc_i_ds(num_of_points: int) -> list[float]:
    i_d_max = - math.sqrt(PEAK_CURRENT[1]**2 / (math.sqrt(2)*(1+1/math.tan(math.radians(PHASE_ADVANCE[1])))))
    i_d = (0, i_d_max)
    i_ds = _range_tuple_to_list(i_d, num_of_points)
    return i_ds

def do_operation_point_variation(
        number_of_datapoints: int = 1000,
) -> Generator[dict[str, float], None, None]:

    a, b, c = mymath.best_factors(number_of_datapoints, k=3)
    shaft_speeds = _range_tuple_to_list(SHAFT_SPEED, a)
    for shaft_speed in shaft_speeds:
        i_qs = _calc_i_qs(b)
        for i_q in i_qs:
            i_ds = _calc_i_ds(c)
            for i_d in i_ds:
                yield {
                    "ShaftSpeed": shaft_speed,
                    "PeakCurrent": (i_q**2 + i_d**2)**0.5,
                    "PhaseAdvance": -math.degrees(math.atan2(i_d, i_q))
                }


def do_winding_variation(number_of_datapoints: int = 100) -> Generator[dict[str, float], None, None]:
    slot_max, pole_max = mymath.best_factors(number_of_datapoints, k=2)

    for slot in range(3, 3*(slot_max + 4), 3):
        for pole in range(2, 2*(pole_max + 3), 2):
            d = math.gcd(slot, pole)

            if slot % (3 * d) == 0:
                value = pole // (3 * d)

                if value >= 1:
                    yield {"Slot_Number": slot, "Pole_Number": pole}


def test_point_generator(number_of_datapoints: int = 100) -> Generator[dict[str, float], None, None]:
    test_point_list: list[dict[str, float]] = []
    for winding_dict, axial_dict, op_dict, radial_dict in zip(
            do_winding_variation(number_of_datapoints=number_of_datapoints),
            do_axial_variation(number_of_datapoints=number_of_datapoints),
            do_operation_point_variation(number_of_datapoints=number_of_datapoints),
            do_radial_variation(number_of_datapoints=number_of_datapoints)
    ):
        test_point_dict = winding_dict | axial_dict | op_dict | radial_dict
        test_point_list.append(test_point_dict)

    for i in range(number_of_datapoints):
        yield test_point_list[i]



