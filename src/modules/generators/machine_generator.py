
from typing import Generator
import numpy as np
from scipy.stats import qmc
import math
from itertools import combinations_with_replacement
from math import isqrt, prod

from modules.generators.map_generator import map_generator
from config.config_mcad_parameters import *


def _find_pole_slot_pairs(
        num_of_sub_designs: int,
        seed: int = 42
) -> np.ndarray:

    if num_of_sub_designs < 1:
        raise ValueError("num_of_sub_designs must be >= 1")

    def best_factors(target: int, k: int) -> tuple:

        def divisors(n: int) -> list:
            _d = set()
            for i in range(1, isqrt(n) + 1):
                if n % i == 0:
                    _d.add(i)
                    _d.add(n // i)
            return sorted(_d)

        divs = [d for d in divisors(target)]

        best = (None,)
        best_score = float("inf")

        for factors in combinations_with_replacement(divs, k):
            if prod(factors) == target:
                score = max(factors) - min(factors)
                if score < best_score:
                    best_score = score
                    best = factors
                    best = tuple(sorted(best, reverse=True))
        return best

    slot_max, pole_max = best_factors(
        num_of_sub_designs,
        k=2
    )

    while True:
        pairs = []

        for slot in range(3, 3 * (slot_max + 4), 3):
            for pole in range(2, 2 * (pole_max + 3), 2):

                d = math.gcd(slot, pole)

                if slot % (3 * d) == 0:
                    value = pole // (3 * d)

                    if value >= 1:
                        pairs.append([slot, pole])

        # genügend gültige Paare gefunden
        if len(pairs) >= num_of_sub_designs:
            break

        # Suchraum vergrößern
        slot_max += 1
        pole_max += 1

    pairs = np.array(pairs, dtype=int)

    # zufällige Auswahl ohne Wiederholung
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)

    return pairs[:num_of_sub_designs]

def _sampling(
        num_of_points: int,
        d: int,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        seed: int
) -> np.ndarray:

    samples = qmc.LatinHypercube(
        d=d,
        optimization="random-cd",
        seed=seed
    ).random(n=num_of_points)

    return qmc.scale(
        samples,
        lower_bounds,
        upper_bounds
    )

def _get_lambda_values(
        num_base_designs: int,
        num_sub_designs: int,
        num_subsub_designs: int,
)-> tuple[np.ndarray, np.ndarray, np.ndarray]:

    lambda_1_samples = _sampling(
        num_of_points=num_base_designs,
        d=3,
        lower_bounds=np.array([0.2, 0.5, 0.01]),
        upper_bounds=np.array([0.8, 0.9, 5.0]),
        seed=42
    )

    lambda_2_samples = _sampling(
        num_of_points=num_sub_designs,
        d=3,
        lower_bounds=np.array([0.3, 0.1, 0.6]),
        upper_bounds=np.array([0.7, 0.4, 1.0]),
        seed=33
    )

    lambda_3_samples = _sampling(
        num_of_points=num_subsub_designs,
        d=3,
        lower_bounds=np.array([0.0, 0.1, 0.1]),
        upper_bounds=np.array([0.95, 1.0, 5.0]),
        seed=11
    )

    return lambda_1_samples, lambda_2_samples, lambda_3_samples

def _lambda_generator(
        num_base_designs: int,
        num_sub_designs: int,
        num_subsub_designs: int,
        num_operation_points_per_dim: int
)-> Generator[dict[str, float], None, None]:

    ##structure
    #   for lambda_1 in lambdas_1:
    #       for lambda_2 in lambdas_2:
    #           for lambda_3 in lambdas_3:
    #               for op in ops:
    #                   yield geometrical design at op
    #
    # -> number_of_datapoints = num_base_designs * num_sub_designs * num_subsub_designs * num_operation_points

    # radial: lambda_1_1 = D_Si / D_Sa
    # tangentail: lambda_1_2 = beta_PM / beta_p = 2 * Magnet_Arc_[ED] / 360
    #       beta_m = Magnet_Arc_[ED] / p ; beta_p = 360° / p
    # axial: lambda_1_3 = l_PM / g;

    # d_emb + h_PM < (D_Sa - D_Si) / 2
    # 2* (d_emb + h_PM) < D_Si * (1/lambda_1 - 1)
    # 1/ (1 + 2 * (d_emb + h_PM) / D_Si) > lambda_1

    # tangential: lambda_2_1 = b_S / tau_s = s*b_S / pi*D_Si
    #   tau_s = pi * D_Si / s = pi * D_Si  / S
    #   b_S = pi * lambda_2_1 * D_Si / s
    # axial: lambda_2_2 = dicke des mag rückflusses / tau_p = 2*p * (l_S - d_s) / (pi * D_Si * (1+1/lambda_1_1))
    #   l_S = d_S + pi * lambda_2_2 * D_Si * (1+1/lambda_1_1) / 2*p
    # radial: lambda_2_3 = h_PM / (D_Sa - D_Si)/2
    #   h_PM = lambda_2_3 * D_Si * (1/lambda_1_1 - 1) / 2

    # radial: lambda_3_1 = d_emb / ((D_Si*(1/lambda_1_1 -1)/2 - h_PM)
    #   d_emb = lambda_3_1 * ((D_Si*(1/lambda_1_1 -1) - h_PM)
    # tangential: lambda_3_2 = 1 / CircumferentialSegments
    # axial: lambda_3_3 = l_R / l_PM
    #   l_R = lambda_3_3 * l_PM = lambda_3_3 * lambda_1_3 * delta


    lambda_1_samples, lambda_2_samples, lambda_3_samples = _get_lambda_values(
        num_base_designs=num_base_designs,
        num_sub_designs=num_sub_designs,
        num_subsub_designs=num_subsub_designs,
    )


    stator_bore_samples = _sampling(
        num_of_points=num_base_designs,
        d=1,
        lower_bounds=np.array([STATOR_BORE[0]]),
        upper_bounds=np.array([STATOR_BORE[1]]),
        seed=68
    )
    airgap_samples = _sampling(
        num_of_points=num_base_designs,
        d=1,
        lower_bounds=np.array([AIRGAP[0]]),
        upper_bounds=np.array([AIRGAP[1]]),
        seed=85
    )
    slot_depth_samples = _sampling(
        num_of_points=num_sub_designs,
        d=1,
        lower_bounds=np.array([SLOT_DEPTH[0]]),
        upper_bounds=np.array([SLOT_DEPTH[1]]),
        seed=10
    )
    slot_pole_samples = _find_pole_slot_pairs(num_of_sub_designs=num_sub_designs)

    op_samples = map_generator(
        i_min_max=SHAFT_SPEED,
        j_min_max=Q_CURRENT_DENSITY,
        k_min_max=D_CURRENT_DENSITY,
        num_operation_points_per_dim=num_operation_points_per_dim
    )


    for (
        (lambda_1_1, lambda_1_2, lambda_1_3),
        (stator_bore_sample,),
        (airgap_sample,)
    ) in zip(lambda_1_samples, stator_bore_samples, airgap_samples):
        for (
            (lambda_2_1, lambda_2_2, lambda_2_3),
            (slot_sample, pole_sample),
            (slot_depth_sample,)
        ) in zip(lambda_2_samples, slot_pole_samples, slot_depth_samples):
            for (lambda_3_1, lambda_3_2, lambda_3_3) in lambda_3_samples:
                for (shaft_speed, q_current_density, d_current_density) in op_samples:


                    yield {
                        "Magnet_Arc_[ED]": lambda_1_2 * 360 / 2,
                        "Stator_Lam_Dia": stator_bore_sample / lambda_1_1,
                        "AFM_D_Rotor": stator_bore_sample / lambda_1_1,
                        "Stator_Bore": stator_bore_sample,
                        "Airgap": airgap_sample,
                        "Magnet_Length": lambda_1_3 * airgap_sample,

                        "Slot_Number": int(slot_sample),
                        "Pole_Number": int(pole_sample),
                        "Slot_Width": math.pi * lambda_2_1 * stator_bore_sample  / slot_sample,
                        "Stator_Lam_Length": slot_depth_sample + math.pi * lambda_2_2 * stator_bore_sample * (1 + 1/lambda_1_1) / (2*pole_sample),
                        "Slot_Depth": slot_depth_sample,
                        "Magnet_Thickness": lambda_2_3 * stator_bore_sample * (1/lambda_1_1 - 1) / 2,

                        "Magnet_Embed_Depth": lambda_3_1 * (stator_bore_sample*(1/lambda_1_1 - 1)/2 * (1-lambda_2_3)),
                        "Rotor_Lam_Length": lambda_3_3 * lambda_1_3 * airgap_sample,
                        "CircumferentialSegments": round(1/lambda_3_2),

                        "Motor_Length": 2*(
                                airgap_sample
                                + slot_depth_sample + math.pi * lambda_2_2 * stator_bore_sample * (1 + 1/lambda_1_1) / (2*pole_sample)
                                + lambda_1_3 * airgap_sample
                        ) + lambda_3_3 * lambda_1_3 * airgap_sample, #2*(g + l_S + l_PM) + l_R

                        "Tooth_Tip_Angle": 30,
                        "Slot_Opening": 0.8* math.pi * lambda_2_1 * stator_bore_sample  / slot_sample,


                        "RMSCurrentDensity": current_density,
                        "ShaftSpeed": shaft_speed,
                        "PhaseAdvance": math.tan(d_current_density / q_current_density),
                    }



def get_generator()-> Generator[dict[str, float], None, None]:
    return _lambda_generator()

