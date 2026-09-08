
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
            num_sub_designs: int,
            num_subsub_designs: int,
            num_operation_points_per_dim: int
    ) -> None:
        self.num_base_designs = num_base_designs
        self.num_sub_designs = num_sub_designs
        self.num_subsub_designs = num_subsub_designs
        self.num_operation_points_per_dim = num_operation_points_per_dim
        self.num_datapoints = num_base_designs * num_sub_designs * num_subsub_designs * num_operation_points_per_dim**3

    def get_generator(self) -> Generator[dict[str, float], None, None]:
        return self._lambda_generator()


    def _find_pole_slot_pairs(self, seed: int = 42) -> np.ndarray:
        """Select well-spread [slot, pole] pairs from slots 3..30 and poles 2..20.

        Match normalized LHS targets to distinct valid pairs at minimum total
        squared distance. The discrete constraints only approximate an LHS.
        """

        num_pairs = self.num_sub_designs
        if num_pairs < 1:
            raise ValueError("num_of_sub_designs must be >= 1")

        pairs = []
        for slot in range(3, 31, 3):
            for pole in range(2, 21, 2):
                three_gcd = 3 * math.gcd(slot, pole)
                if slot % three_gcd == 0:
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
            num_of_points=self.num_base_designs,
            d=3,
            lower_bounds=np.array([0.2, 0.5, 0.01]),
            upper_bounds=np.array([0.8, 0.9, 5.0]),
            seed=42
        )

        lambda_2_samples = self._sampling(
            num_of_points=self.num_sub_designs,
            d=3,
            lower_bounds=np.array([0.3, 0.1, 0.6]),
            upper_bounds=np.array([0.7, 0.4, 1.0]),
            seed=33
        )

        lambda_3_samples = self._sampling(
            num_of_points=self.num_subsub_designs,
            d=3,
            lower_bounds=np.array([0.0, 0.1, 0.1]),
            upper_bounds=np.array([0.95, 1.0, 5.0]),
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
        slot_depth_samples = self._sampling(
            num_of_points=self.num_sub_designs,
            d=1,
            lower_bounds=np.array([SLOT_DEPTH[0]]),
            upper_bounds=np.array([SLOT_DEPTH[1]]),
            seed=10
        )
        slot_pole_samples = self._find_pole_slot_pairs()

        op_samples = map_generator(
            i_min_max=SHAFT_SPEED,
            j_min_max=Q_CURRENT_DENSITY,
            k_min_max=D_CURRENT_DENSITY,
            num_operation_points_per_dim=self.num_operation_points_per_dim
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