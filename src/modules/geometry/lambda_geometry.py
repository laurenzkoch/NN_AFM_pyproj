from __future__ import annotations

import math
from dataclasses import Field, dataclass, field, fields

import numpy as np
from numpy.typing import NDArray

@dataclass(frozen=True, slots=True)
class LambdaDefinition:
    column: str
    formula: str
    sampling_bounds: tuple[float, float]


def _lambda_field(
    column: str,
    formula: str,
    sampling_bounds: tuple[float, float],
) -> Field[float]:
    lower, upper = sampling_bounds

    if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
        raise ValueError(f"Invalid sampling bounds for {column}.")

    return field(
        metadata={
            "definition": LambdaDefinition(
                column=column,
                formula=formula,
                sampling_bounds=sampling_bounds,
            ),
        },
    )

@dataclass(frozen=True, slots=True, kw_only=True)
class LambdaValues:
    lambda_1_diameter_ratio: float = _lambda_field(
        "lambda_1", "D_Si / D_Sa", (0.2, 0.8),
    )
    lambda_2_magnet_arc_ratio: float = _lambda_field(
        "lambda_2", "Magnet_Arc_[ED] / 180", (0.5, 0.9),
    )
    lambda_3_airgap_magnet_length_ratio: float = _lambda_field(
        "lambda_3", "g / l_PM", (0.2, 1.0),
    )
    lambda_4_slot_width_pitch_ratio: float = _lambda_field(
        "lambda_4", "s * b_S / (pi * D_Si)", (0.3, 0.7),
    )
    lambda_5_slot_width_depth_ratio: float = _lambda_field(
        "lambda_5", "b_S / d_S", (0.1, 1.0),
    )
    lambda_6_magnet_radial_ratio: float = _lambda_field(
        "lambda_6", "h_PM / ((D_Sa - D_Si) / 2)", (0.6, 1.0),
    )
    lambda_7_embed_remaining_width_ratio: float = _lambda_field(
        "lambda_7", "d_emb / ((D_Sa - D_Si) / 2 - h_PM)", (0.0, 0.95),
    )
    lambda_8_stator_back_iron_pitch_ratio: float = _lambda_field(
        "lambda_8", "2 * p * (l_S - d_S) / (pi * (D_Sa + D_Si))", (0.1, 0.4),
    )
    lambda_9_magnet_rotor_length_ratio: float = _lambda_field(
        "lambda_9", "l_PM / l_R", (0.2, 1.0),
    )
    lambda_10_slot_opening_width_ratio: float = _lambda_field(
        "lambda_10", "b_tto / b_S", (0.1, 1.0),
    )
    lambda_11_tooth_tip_depth_ratio: float = _lambda_field(
        "lambda_11", "d_ttd / d_S", (0.0, 0.3),
    )
    lambda_12_slot_corner_radius_ratio: float = _lambda_field(
        "lambda_12", "r_S / (b_S / 2)", (0.0, 1.0),
    )

    def __post_init__(self) -> None:
        for item in fields(self):
            value: float = getattr(self, item.name)

            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f"{item.name} must be a real number.")

            if not math.isfinite(value):
                raise ValueError(f"{item.name} must be finite.")

            # Store ordinary Python floats consistently.
            object.__setattr__(self, item.name, float(value))

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(item.name for item in fields(cls))

    @classmethod
    def from_array(cls, values: NDArray[np.float64]) -> LambdaValues:

        names: tuple[str, ...] = cls.names()
        if values.ndim != 1 or values.shape[0] != len(names):
            raise ValueError(
                f"Expected a one-dimensional array with {len(names)} values, "
                f"got shape {values.shape}."
            )

        return cls(
            **{name: float(value) for name, value in zip(names, values, strict=True)}
        )

    def to_array(self) -> NDArray[np.float64]:

        return np.asarray(
            [getattr(self, name) for name in self.names()],
            dtype=np.float64,
        )

    @classmethod
    def definitions(cls) -> tuple[LambdaDefinition, ...]:
        definitions: list[LambdaDefinition] = []

        for item in fields(cls):
            definition = item.metadata.get("definition")

            if not isinstance(definition, LambdaDefinition):
                raise TypeError(f"Missing lambda definition: {item.name}")

            definitions.append(definition)

        columns = [definition.column for definition in definitions]
        if len(columns) != len(set(columns)):
            raise ValueError("Lambda column names must be unique.")

        return tuple(definitions)

    @classmethod
    def columns(cls) -> list[str]:
        return [definition.column for definition in cls.definitions()]

    @classmethod
    def sampling_bounds(cls) -> list[tuple[float, float]]:
        return [
            definition.sampling_bounds
            for definition in cls.definitions()
        ]