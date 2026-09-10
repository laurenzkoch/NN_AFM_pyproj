
from typing import Generator
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import qmc
import math

from src.modules.generators.map_generator import map_generator
from src.config.config_mcad_parameters import *
from src.modules.process.motor_cad_processes.mcad_processing import test_boundary_conditions


class AFMMachineGenerator:

    def __init__(
            self,
            num_base_designs: int,
            num_lam_designs: int,
            num_operation_points_per_dim: int,
            num_data_points: int,
    ) -> None:
        self.num_base_designs = num_base_designs
        self.num_lam_designs = num_lam_designs
        self.num_operation_points_per_dim = num_operation_points_per_dim
        self.num_datapoints = num_data_points

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def get_generator(self) -> Generator[dict[str, float], None, None]:
        return self._lambda_generator()


    def _find_pole_slot_pairs(self, seed: int = 42) -> np.ndarray:
        """Select well-spread [slot, pole] pairs from slots 3..30 and poles 2..20.

        Match normalized LHS targets to distinct valid pairs at minimum total
        squared distance. The discrete constraints only approximate an LHS.
        """

        num_pairs = self.num_base_designs
        if num_pairs < 1:
            raise ValueError("num_of_sub_designs must be >= 1")

        pairs = []
        for slot in range(3, 31, 3):
            for pole in range(2, 21, 2):
                gcd = math.gcd(slot, pole)
                if slot % (3*gcd) == 0 and slot / gcd <= 6:
                    pairs.append((slot, pole))

        if num_pairs > len(pairs):
            raise ValueError(
                f"num_sub_designs must be <= {len(pairs)} "
                "for slots 3..30 and poles 2..20"
            )

        pairs = np.array(pairs, dtype=int)
        # Beide Achsen gleich gewichten, unabhängig von ihren Wertebereichen.
        normalized_pairs = (pairs - np.array([3, 2])) / np.array([27, 18])
        targets = self._sampling(
            num_of_points=num_pairs,
            d=2,
            lower_bounds=np.zeros(2),
            upper_bounds=np.ones(2),
            seed=seed,
        )

        # Gemeinsame Zuordnung verhindert doppelte Paare beim Annähern der Ziele.
        squared_distances = np.sum(
            (targets[:, None, :] - normalized_pairs[None, :, :]) ** 2,
            axis=2,
        )
        _, selected_indices = linear_sum_assignment(squared_distances)
        return pairs[selected_indices]

    @staticmethod
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

    def _get_lambda_values(self)-> tuple[np.ndarray, np.ndarray, np.ndarray]:

        lambda_1_samples = self._sampling(
            num_of_points=self.num_lam_designs,
            d=3,
            lower_bounds=np.array([LAMBDA_1_1[0], LAMBDA_1_2[0], LAMBDA_1_3[0]]),
            upper_bounds=np.array([LAMBDA_1_1[1], LAMBDA_1_2[1], LAMBDA_1_3[1]]),
            seed=42
        )

        lambda_2_samples = self._sampling(
            num_of_points=self.num_lam_designs,
            d=3,
            lower_bounds=np.array([LAMBDA_2_1[0], LAMBDA_2_2[0], LAMBDA_2_3[0]]),
            upper_bounds=np.array([LAMBDA_2_1[1], LAMBDA_2_2[1], LAMBDA_2_3[1]]),
            seed=33
        )

        lambda_3_samples = self._sampling(
            num_of_points=self.num_lam_designs,
            d=3,
            lower_bounds=np.array([LAMBDA_3_1[0], LAMBDA_3_2[0], LAMBDA_3_3[0]]),
            upper_bounds=np.array([LAMBDA_3_1[1], LAMBDA_3_2[1], LAMBDA_3_3[1]]),
            seed=11
        )

        return lambda_1_samples, lambda_2_samples, lambda_3_samples

    def _lambda_generator(self)-> Generator[dict[str, float], None, None]:

        ##structure
        #   for lambda_1 in lambdas_1:
        #       for lambda_2 in lambdas_2:
        #           for lambda_3 in lambdas_3:
        #               for op in ops:
        #                   yield geometrical design at op
        #
        # -> number_of_datapoints = num_base_designs * num_lam_designs**3 * num_operation_points**3

        # radial: lambda_1_1 = D_Si / D_Sa
        # tangential: lambda_1_2 = beta_PM / beta_p = 2 * Magnet_Arc_[ED] / 360
        #       beta_m = Magnet_Arc_[ED] / p ; beta_p = 360° / p
        # axial: lambda_1_3 = l_PM / g;

        # d_emb + h_PM < (D_Sa - D_Si) / 2
        # 2* (d_emb + h_PM) < D_Si * (1/lambda_1 - 1)
        # 1/ (1 + 2 * (d_emb + h_PM) / D_Si) > lambda_1

        # tangential: lambda_2_1 = b_S / tau_s = s*b_S / pi*D_Si
        #   tau_s = pi * D_Si / s = pi * D_Si  / S
        #   b_S = pi * lambda_2_1 * D_Si / s
        # axial: lambda_2_2 = b_S / d_S
        #       d_S = b_S / lambda_2_2 = pi * lambda_2_1 / lambda_2_2 * D_Si / s
        # radial: lambda_2_3 = h_PM / (D_Sa - D_Si)/2
        #   h_PM = lambda_2_3 * D_Si * (1/lambda_1_1 - 1) / 2

        # radial: lambda_3_1 = d_emb / ((D_Si*(1/lambda_1_1 -1)/2 - h_PM)
        #   d_emb = lambda_3_1 * ((D_Si*(1/lambda_1_1 -1) - h_PM)
        # tangential: lambda_3_2 = dicke des mag rückflusses / tau_p = 2*p * (l_S - d_s) / (pi * D_Si * (1+1/lambda_1_1))
        #         #   l_S = d_S + pi * lambda_3_2 * D_Si * (1+1/lambda_1_1) / 2*p
        # axial: lambda_3_3 = l_R / l_PM
        #   l_R = lambda_3_3 * l_PM = lambda_3_3 * lambda_1_3 * delta


        lambda_1_samples, lambda_2_samples, lambda_3_samples = self._get_lambda_values()


        stator_bore_samples = self._sampling(
            num_of_points=self.num_base_designs,
            d=1,
            lower_bounds=np.array([STATOR_BORE[0]]),
            upper_bounds=np.array([STATOR_BORE[1]]),
            seed=68
        )
        airgap_samples = self._sampling(
            num_of_points=self.num_base_designs,
            d=1,
            lower_bounds=np.array([AIRGAP[0]]),
            upper_bounds=np.array([AIRGAP[1]]),
            seed=85
        )
        slot_pole_samples = self._find_pole_slot_pairs()

        op_samples = map_generator(
            i_min_max=SHAFT_SPEED,
            j_min_max=Q_CURRENT_DENSITY,
            k_min_max=D_CURRENT_DENSITY,
            num_operation_points_per_dim=self.num_operation_points_per_dim
        )

        for (
                (airgap_sample,),
                (slot_sample, pole_sample),
                (stator_bore_sample,),
        ) in zip(airgap_samples, slot_pole_samples, stator_bore_samples):
            for (lambda_1_1, lambda_1_2, lambda_1_3) in lambda_1_samples:
                for (lambda_2_1, lambda_2_2, lambda_2_3) in lambda_2_samples:
                    for (lambda_3_1, lambda_3_2, lambda_3_3) in lambda_3_samples:
                        for (shaft_speed, q_current_density, d_current_density) in op_samples:

                            s = int(slot_sample)
                            p = int(pole_sample)
                            g = airgap_sample
                            d_si = stator_bore_sample
                            d_sa = d_si / lambda_1_1
                            beta_pm = lambda_1_2 * 360 / 2
                            l_pm = lambda_1_3 * g
                            b_s = lambda_2_1 * math.pi * d_si / s
                            t_s = b_s / lambda_2_2
                            b_tto = 0.8 * b_s
                            h_pm = lambda_2_3 * (d_sa - d_si) / 2
                            d_emb = lambda_3_1 * ((d_sa - d_si) / 2 - h_pm)
                            l_r = lambda_3_3 * l_pm
                            d_mean = (d_sa + d_si) / 2
                            l_s = t_s + lambda_3_2 * math.pi * d_mean / p


                            yield {
                                "Slot_Number": s,
                                "Pole_Number": p,
                                "Airgap": g,
                                "Stator_Bore": d_si,

                                "Stator_Lam_Dia": d_sa,
                                "AFM_D_Rotor": d_sa,
                                "Magnet_Arc_[ED]": beta_pm,
                                "Magnet_Length": l_pm,

                                "Slot_Width": b_s,
                                "Slot_Depth": t_s,
                                "Slot_Opening": b_tto,
                                "Magnet_Thickness": h_pm,

                                "Magnet_Embed_Depth": d_emb,
                                "Rotor_Lam_Length": l_r,
                                "Stator_Lam_Length": l_s,

                                "Motor_Length": 2*(g + l_s + l_pm) + l_r,

                                "RMSCurrentDensity": math.sqrt(q_current_density**2 + d_current_density**2) / 2**0.5,
                                "ShaftSpeed": shaft_speed,
                                "PhaseAdvance": math.degrees(math.atan2(-d_current_density, q_current_density)),
                            }

    def geometry_test(self) -> None:
        design_counter = 0
        drop_counter = 0

        for design in self.get_generator():

            design_counter += 1

            default_values = DEFAULT_VALUES.copy()
            for key, value in design.items():
                default_values[key] = value

            parameter_name: list[str] = list(default_values.keys())
            parameter_values: list[float] = list(default_values.values())

            if not test_boundary_conditions(parameter_name, parameter_values):
                print("________TEST FAILED FOR:____________")
                for _var, _value in default_values.items():
                    print(f"{_var}: {_value}")
                drop_counter += 1

        print(f"#############TEST COMPLETED: {drop_counter} from {design_counter} designs failed ##################")
