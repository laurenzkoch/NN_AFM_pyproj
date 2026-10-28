import math
from collections.abc import Generator
from typing import Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

from src.config.config_mcad_parameters import (
    AIRGAP,
    LAMBDA_BOUNDS,
    POLE_NUMBER,
    SLOT_NUMBER,
    STATOR_BORE,
)
from src.modules.geometry.lambda_geometry import (
    LambdaValues,
    geometry_from_lambdas,
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

    def _get_lambda_values_v2(self) -> tuple[LambdaValues, ...]:
        columns: list[str] = LambdaValues.columns()

        missing = set(columns) - set(LAMBDA_BOUNDS)
        unexpected = set(LAMBDA_BOUNDS) - set(columns)

        if missing or unexpected:
            raise ValueError(
                "Lambda sampling configuration differs from definitions: "
                f"missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

        bounds: NDArray[np.float64] = np.asarray(
            [LAMBDA_BOUNDS[column] for column in columns],
            dtype=np.float64,
        )

        if bounds.shape != (len(columns), 2):
            raise ValueError("Each lambda must have two sampling bounds.")

        if not bool(np.isfinite(bounds).all()):
            raise ValueError("Lambda sampling bounds must be finite.")

        if bool((bounds[:, 0] >= bounds[:, 1]).any()):
            raise ValueError(
                "Each lambda lower bound must be smaller than its upper bound."
            )

        samples: NDArray[np.float64] = self._sampling(
            num_of_points=self.num_lam_designs,
            d=len(columns),
            lower_bounds=bounds[:, 0],
            upper_bounds=bounds[:, 1],
            seed=42,
        )

        return tuple(LambdaValues.from_array(sample) for sample in samples)

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
        lambdas: LambdaValues,
        shaft_speed_norm: float,
        q_current_density: float,
        d_current_density: float,
    ) -> dict[str, float]:


        geometry: dict[str, float] = geometry_from_lambdas(
            lambdas,
            airgap=airgap_sample,
            stator_bore=stator_bore_sample,
            slot_number=SLOT_NUMBER,
            pole_number=POLE_NUMBER,
        )

        geometry.update(
            {
                "RMSCurrentDensity": math.hypot(
                    q_current_density,
                    d_current_density,
                ),
                # shaft_speed_norm = shaft_speed / (d_sa*pi)
                "ShaftSpeed": (shaft_speed_norm * geometry["Stator_Lam_Dia"] * math.pi),
                "PhaseAdvance": math.degrees(
                    math.atan2(-d_current_density, q_current_density)
                ),
            }
        )

        return geometry

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
