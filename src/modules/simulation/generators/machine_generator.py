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
    ) -> None:
        for name, value in (
                ("num_base_designs", num_base_designs),
                ("num_lam_designs", num_lam_designs),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

        self.num_base_designs = num_base_designs
        self.num_lam_designs = num_lam_designs

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

    def _get_base_samples(self) -> NDArray[np.float64]:
        return self._sampling(
            num_of_points=self.num_base_designs,
            d=2,
            lower_bounds=np.array([AIRGAP[0], STATOR_BORE[0]]),
            upper_bounds=np.array([AIRGAP[1], STATOR_BORE[1]]),
            seed=17,
        )

    def _lambda_generator_v2(self) -> Generator[dict[str, float]]:
        lambda_sets = self._get_lambda_values_v2()
        base_samples = self._get_base_samples()

        for airgap_sample, stator_bore_sample in base_samples:
            for lambda_set in lambda_sets:
                yield geometry_from_lambdas(
                    lambda_set,
                    airgap=float(airgap_sample),
                    stator_bore=float(stator_bore_sample),
                    slot_number=SLOT_NUMBER,
                    pole_number=POLE_NUMBER,
                )
