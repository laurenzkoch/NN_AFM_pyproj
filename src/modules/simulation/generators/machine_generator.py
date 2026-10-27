import math
from collections.abc import Generator
from typing import Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

from src.config.config_mcad_parameters import (
    AIRGAP,
    LAMBDA_MIN_MAX,
    POLE_NUMBER,
    SLOT_NUMBER,
    STATOR_BORE,
)
from src.modules.simulation.generators.map_generator import map_generator


class _LegacyLatinHypercubeFactory(Protocol):
    def __call__(
        self,
        *,
        d: int,
        optimization: Literal["random-cd"],
        seed: int,
    ) -> qmc.LatinHypercube: ...


class AFMMachineGenerator:
    def __init__(
        self,
        num_base_designs: int,
        num_lam_designs: int,
        num_data_points: int,
    ) -> None:
        self.num_base_designs = num_base_designs
        self.num_lam_designs = num_lam_designs
        self.num_datapoints = num_data_points

    def get_generator(self, _type: str) -> Generator[dict[str, float]]:
        match _type:
            case "v1":
                raise ValueError("Type v1 is no longer supported")
            case "v2":
                return self._lambda_generator_v2()
            case _:
                raise ValueError(f"Unknown generator type: {_type}")

    @staticmethod
    def _sampling(
        num_of_points: int,
        d: int,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        factory: _LegacyLatinHypercubeFactory = cast(
            _LegacyLatinHypercubeFactory,
            cast(object, qmc.LatinHypercube),
        )

        sampler: qmc.LatinHypercube = factory(
            d=d,
            optimization="random-cd",
            seed=seed,
        )
        samples: NDArray[np.float64] = sampler.random(n=num_of_points)

        return qmc.scale(samples, lower_bounds, upper_bounds)

    def _get_lambda_values_v2(self) -> np.ndarray:

        lambda_min_max: np.ndarray = np.array(
            LAMBDA_MIN_MAX,
            dtype=np.float64,
        )
        mins = lambda_min_max[:, 0]
        maxs = lambda_min_max[:, 1]

        lambdas: np.ndarray = self._sampling(
            num_of_points=self.num_lam_designs,
            d=len(LAMBDA_MIN_MAX),
            lower_bounds=mins,
            upper_bounds=maxs,
            seed=42,
        )

        return lambdas

    def _get_samples(self) -> tuple[np.ndarray, np.ndarray]:

        airgap_dsi_samples: np.ndarray = self._sampling(
            num_of_points=self.num_base_designs,
            d=2,
            lower_bounds=np.array([AIRGAP[0], STATOR_BORE[0]]),
            upper_bounds=np.array([AIRGAP[1], STATOR_BORE[1]]),
            seed=17,
        )

        op_samples: np.ndarray = map_generator()

        return airgap_dsi_samples, op_samples

    @staticmethod
    def _calculate_geometric(
        airgap_sample: float,
        stator_bore_sample: float,
        lambdas: NDArray[np.float64],
        shaft_speed_norm: float,
        q_current_density: float,
        d_current_density: float,
    ) -> dict[str, float]:

        # radial: lambda_1 = D_Si / D_Sa

        # tangential: lambda_2 = beta_PM / beta_p = 2 * Magnet_Arc_[ED] / 360
        #       beta_m = Magnet_Arc_[ED] / p ; beta_p = 360° / p

        # axial: lambda_3 = g / l_PM;

        # d_emb + h_PM < (D_Sa - D_Si) / 2
        # 2* (d_emb + h_PM) < D_Si * (1/lambda_1 - 1)
        # 1/ (1 + 2 * (d_emb + h_PM) / D_Si) > lambda_1

        # tangential: lambda_2_1 = b_S / tau_s = s*b_S / pi*D_Si
        #   tau_s = pi * D_Si / s = pi * D_Si  / S
        #   b_S = pi * lambda_4 * D_Si / s

        # axial: lambda_5 = b_S / d_S
        #       d_S = b_S / lambda_5 = pi * lambda_4 / lambda_5 * D_Si / s

        # radial: lambda_6 = h_PM / (D_Sa - D_Si)/2
        #   h_PM = lambda_6 * D_Si * (1/lambda_1 - 1) / 2

        # radial: lambda_7 = d_emb / ((D_Si*(1/lambda_1 -1)/2 - h_PM)
        #   d_emb = lambda_7 * ((D_Si*(1/lambda_1 -1) - h_PM)

        # tangential: lambda_8 = Dicke des magnetischen Rückflusses / tau_p
        # = 2*p * (l_S - d_s) / (pi * D_Si * (1 + 1/lambda_1))
        #         #   l_S = d_S + pi * lambda_8 * D_Si * (1+1/lambda_1) / 2*p

        # axial: lambda_9 = l_PM / l_R
        #   l_R =  l_PM / lambda_9

        # lambda_10 = b_tto / b_S

        # lambda_11 = d_ttd / d_S

        # shaft_speed_norm = shaft_speed / d_sa
        #w*d_a/2 = 2*pi*n*d_a/2 = v_umfang [mm / min]

        s = SLOT_NUMBER
        p = POLE_NUMBER
        g = airgap_sample
        d_si = stator_bore_sample
        d_sa = d_si / lambdas[0]
        beta_pm = lambdas[1] * 360 / 2
        l_pm = g / lambdas[2]
        b_s = lambdas[3] * math.pi * d_si / s
        d_s = b_s / lambdas[4]
        h_pm = lambdas[5] * (d_sa - d_si) / 2
        d_emb = lambdas[6] * ((d_sa - d_si) / 2 - h_pm)
        d_mean = (d_sa + d_si) / 2
        l_s = d_s + lambdas[7] * math.pi * d_mean / p
        l_r = l_pm / lambdas[8]
        b_tto = lambdas[9] * b_s
        d_ttd = lambdas[10] * d_s
        r_s = lambdas[11] * (b_s/2)

        return {
            "Slot_Number": s,
            "Pole_Number": p,
            "Airgap": g,
            "Stator_Bore": d_si,
            "Stator_Lam_Dia": d_sa,
            "AFM_D_Rotor": d_sa,
            "Magnet_Arc_[ED]": beta_pm,
            "Magnet_Length": l_pm,
            "Slot_Width": b_s,
            "Slot_Depth": d_s,
            "Slot_Opening": b_tto,
            "Slot_Corner_Radius": r_s,
            "Tooth_Tip_Depth": d_ttd,
            "Magnet_Thickness": h_pm,
            "Magnet_Embed_Depth": d_emb,
            "Rotor_Lam_Length": l_r,
            "Stator_Lam_Length": l_s,
            "Motor_Length": 2 * (g + l_s + l_pm) + l_r,
            "RMSCurrentDensity": math.sqrt(q_current_density**2 + d_current_density**2),
            "ShaftSpeed": shaft_speed_norm * d_sa,
            "PhaseAdvance": math.degrees(
                math.atan2(-d_current_density, q_current_density)
            ),
        }

    def _lambda_generator_v2(self) -> Generator[dict[str, float]]:
        lambdas = self._get_lambda_values_v2()
        airgap_dsi_samples, op_samples = self._get_samples()

        for airgap_sample, stator_bore_sample in airgap_dsi_samples:
            for lambda_set in lambdas:
                for (
                    shaft_speed_norm,
                    q_current_density,
                    d_current_density,
                ) in op_samples:
                    yield self._calculate_geometric(
                        airgap_sample,
                        stator_bore_sample,
                        lambda_set,
                        shaft_speed_norm,
                        q_current_density,
                        d_current_density,
                    )
