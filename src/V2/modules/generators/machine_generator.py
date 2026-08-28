
from typing import Generator
import numpy as np
from scipy.stats import qmc
import math
from V2.modules.utilitis import mymath

from V2.config.config_mcad_parameters import *

def _find_pole_slot_pairs(num_of_sub_designs: int) -> np.ndarray:
    slot_max, pole_max = mymath.best_factors(num_of_sub_designs, k=2)

    pairs = []

    for slot in range(3, 3*(slot_max + 4), 3):
        for pole in range(2, 2*(pole_max + 3), 2):
            d = math.gcd(slot, pole)

            if slot % (3 * d) == 0:
                value = pole // (3 * d)

                if value >= 1:
                    pairs.append([slot, pole])

    return np.array(pairs, dtype=int)

def _sampling(
        num_of_designs: int,
        d: int,
        seed: int,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray
) -> Generator[np.ndarray, None, None]:

    samples = qmc.LatinHypercube(
        d=d,
        optimization="random-cd",
        seed=seed
    ).random(n=num_of_designs)

    scaled_samples = qmc.scale(
        samples,
        lower_bounds,
        upper_bounds
    )

    for scaled_sample in scaled_samples:
        yield scaled_sample


def _lambda_generator(
        num_base_designs: int,
        num_sub_designs: int,
        num_subsub_designs: int,
        num_operation_points: int
)-> Generator[dict[str, float], None, None]:

    ##structure
    #   lambda_1 definition: num_base_designs
    #   for lambda_1 in lambdas_1:
    #       lambda_2 defintion: num_sub_designs
    #       for lambda_2 in lambdas_2:
    #           lambda_3 definition: num_subsub_designs
    #           for lambda_3 in lambdas_3:
    #               operation points defnition: num_operation_points
    #               for op in ops:
    #                   yield geometrical design at op
    #
    # -> number_of_datapoints = num_base_designs * num_sub_designs * num_subsub_designs * num_operation_points


    #radial: lambda_1_1 = D_Si / D_Sa
    #tangentail: lambda_1_2 = beta_PM / beta_p = 2 * Magnet_Arc_[ED] / 360
    #       beta_m = Magnet_Arc_[ED] / p ; beta_p = 360° / p
    #axial: lambda_1_3 = l_PM / delta;

    # d_emb + h_PM < (D_Sa - D_Si) / 2
    # 2* (d_emb + h_PM) < D_Si * (1/lambda_1 - 1)
    # 1/ (1 + 2 * (d_emb + h_PM) / D_Si) > lambda_1

    lambda_1_samples = _sampling(
        num_of_designs=num_base_designs,
        d=3,
        seed=42,
        lower_bounds=np.array([0.1, 0.1, 0.1]), #[lambda_1_1, lambda_1_2, lambda_1_3]
        upper_bounds=np.array([1.0, 1.0, 1.0]) #[lambda_1_1, lambda_1_2, lambda_1_3]
    )
    stator_bore_samples = _sampling(
        num_of_designs=num_base_designs,
        d=1,
        seed=68,
        lower_bounds=np.array([STATOR_BORE[0]]),
        upper_bounds=np.array([STATOR_BORE[1]])
    )
    airgap_samples = _sampling(
        num_of_designs=num_base_designs,
        d=1,
        seed=85,
        lower_bounds=np.array([AIRGAP[0]]),
        upper_bounds=np.array([AIRGAP[1]])
    )


    for (
        (lambda_1_1, lambda_1_2, lambda_1_3),
        (stator_bore_sample,),
        (airgap_sample,)
    ) in zip(
        lambda_1_samples,
        stator_bore_samples,
        airgap_samples
    ):
        #tangential: lambda_2_1 = b_S / tau_s = 2*s*b_S / pi*D_Si * (1+1/lambda_1_1)
            # tau_s = pi * (D_Si+D_Sa) / 2*s = pi * D_Si * (1+1/lambda_1_1) / 2*S
            # b_S = pi * lambda_2_1 * D_Si * (1+1/lambda_1_1) / 2*s
        #axial: lambda_2_2 = dicke des mag rückflusses / tau_p = 2*p * (l_S - d_s) / (pi * D_Si * (1+1/lambda_1_1))
            # l_S = d_S + pi * lambda_2_2 * D_Si * (1+1/lambda_1_1) / 2*p
        #radial: lambda_2_3 = h_PM / (D_Sa - D_Si)/2
            # h_PM = lambda_2_3 * D_Si * (1/lambda_1_1 - 1) / 2


        lambda_2_samples = _sampling(
            num_of_designs=num_sub_designs,
            d=3,
            seed=33,
            lower_bounds=np.array([0.1, 0.1, 0.1]), #[lambda_2_1, lambda_2_2, lambda_2_3]
            upper_bounds=np.array([0.9, 1.0, 1.0]) #[lambda_2_1, lambda_2_2, lambda_2_3]
        )
        slot_depth_samples = _sampling(
            num_of_designs=num_sub_designs,
            d=1,
            seed=10,
            lower_bounds=np.array([SLOT_DEPTH[0]]),
            upper_bounds=np.array([SLOT_DEPTH[1]])
        )

        slot_pole_samples = _find_pole_slot_pairs(num_of_sub_designs=num_sub_designs)
        rng = np.random.default_rng(42)
        rng.shuffle(slot_pole_samples)


        for (
            (lambda_2_1, lambda_2_2, lambda_2_3),
            (slot_sample, pole_sample),
            (slot_depth_sample,)
        ) in zip(
            lambda_2_samples,
            slot_pole_samples,
            slot_depth_samples
        ):

            # radial: lambda_3_1 = d_emb / ((D_Si*(1/lambda_1_1 -1)/2 - h_PM)
                # d_emb = lambda_3_1 * ((D_Si*(1/lambda_1_1 -1) - h_PM)
            # tangential: lambda_3_2 = CircumferentialSegments
                # -> dimensionslose, diskrete Variable
            # alternativ: lambda_3_2 = 1 / CircumferentialSegments
            # axial: lambda_3_3 = l_R / l_PM
                # l_R = lambda_3_3 * l_PM = lambda_3_3 * lambda_1_3 * delta

            lambda_3_samples = _sampling(
                num_of_designs=num_subsub_designs,
                d=3,
                seed=11,
                lower_bounds=np.array([0.1, 0.1, 0.1]),  # [lambda_3_1, lambda_3_2, lambda_3_3]
                upper_bounds=np.array([1.0, 1.0, 1.0])  # [lambda_3_1, lambda_3_2, lambda_3_3]
            )


            for (lambda_3_1, lambda_3_2, lambda_3_3) in lambda_3_samples:

                op_samples = _sampling(
                    num_of_designs=num_operation_points,
                    d=3,
                    seed=42,
                    lower_bounds=np.array([SHAFT_SPEED[0], PHASE_ADVANCE[0], PEAK_CURRENT[0]]),
                    upper_bounds=np.array([SHAFT_SPEED[1], PHASE_ADVANCE[1], PEAK_CURRENT[1]])
                )
                for shaft_speed, phase_advance, peak_current in op_samples:

                    yield {
                        "Magnet_Arc_[ED]": lambda_1_2 * 360 / 2,
                        "Stator_Lam_Dia": stator_bore_sample / lambda_1_1,
                        "AFM_D_Rotor": stator_bore_sample / lambda_1_1,
                        "Stator_Bore": stator_bore_sample,
                        "Airgap": airgap_sample,
                        "Magnet_Length": lambda_1_3 * airgap_sample,

                        "Slot_Number": int(slot_sample),
                        "Pole_Number": int(pole_sample),
                        "Slot_Width": math.pi * lambda_2_1 * stator_bore_sample * (1 + 1/lambda_1_1) / (2*slot_sample),
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
                        ) + lambda_3_3 * lambda_1_3 * airgap_sample,


                        "PeakCurrent": peak_current,
                        "ShaftSpeed": shaft_speed,
                        "PhaseAdvance": phase_advance,
                    }

def get_generator(
        num_base_designs: int,
        num_sub_designs: int,
        num_subsub_designs: int,
        num_operation_points: int
)-> Generator[dict[str, float], None, None]:
    return _lambda_generator(num_base_designs, num_sub_designs, num_subsub_designs, num_operation_points)

