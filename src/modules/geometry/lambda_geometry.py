from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields

import numpy as np
from numpy.typing import NDArray

GEOMETRY_INPUT_COLUMNS: tuple[str, ...] = (
    "Slot_Number",
    "Pole_Number",
    "Airgap",
    "Stator_Bore",
    "Stator_Lam_Dia",
    "Magnet_Arc_[ED]",
    "Magnet_Length",
    "Slot_Width",
    "Slot_Depth",
    "Magnet_Thickness",
    "Magnet_Embed_Depth",
    "Stator_Lam_Length",
    "Rotor_Lam_Length",
    "Slot_Opening",
    "Tooth_Tip_Depth",
    "Slot_Corner_Radius",
)


@dataclass(frozen=True, slots=True)
class LambdaDefinition:
    column: str
    formula: str


def _lambda_metadata(
    column: str,
    formula: str,
) -> dict[str, LambdaDefinition]:
    return {
        "definition": LambdaDefinition(
            column=column,
            formula=formula,
        ),
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class LambdaValues:
    lambda_1_diameter_ratio: float = field(
        metadata=_lambda_metadata(
            "lambda_1",
            "D_Si / D_Sa",
        ),
    )
    lambda_2_magnet_arc_ratio: float = field(
        metadata=_lambda_metadata(
            "lambda_2",
            "Magnet_Arc_[ED] / 180",
        ),
    )
    lambda_3_airgap_magnet_length_ratio: float = field(
        metadata=_lambda_metadata(
            "lambda_3",
            "g / l_PM",
        ),
    )
    lambda_4_slot_width_pitch_ratio: float = field(
        metadata=_lambda_metadata(
            "lambda_4",
            "s * b_S / (pi * D_Si)",
        ),
    )
    lambda_5_slot_width_depth_ratio: float = field(
        metadata=_lambda_metadata(
            "lambda_5",
            "b_S / d_S",
        ),
    )
    lambda_6_magnet_radial_ratio: float = field(
        metadata=_lambda_metadata(
            "lambda_6",
            "h_PM / ((D_Sa - D_Si) / 2)",
        ),
    )
    lambda_7_embed_remaining_width_ratio: float = field(
        metadata=_lambda_metadata(
            "lambda_7",
            "d_emb / ((D_Sa - D_Si) / 2 - h_PM)",
        ),
    )
    lambda_8_stator_back_iron_pitch_ratio: float = field(
        metadata=_lambda_metadata(
            "lambda_8",
            "2 * p * (l_S - d_S) / (pi * (D_Sa + D_Si))",
        ),
    )
    lambda_9_magnet_rotor_length_ratio: float = field(
        metadata=_lambda_metadata(
            "lambda_9",
            "l_PM / l_R",
        ),
    )
    lambda_10_slot_opening_width_ratio: float = field(
        metadata=_lambda_metadata(
            "lambda_10",
            "b_tto / b_S",
        ),
    )
    lambda_11_tooth_tip_depth_ratio: float = field(
        metadata=_lambda_metadata(
            "lambda_11",
            "d_ttd / d_S",
        ),
    )
    lambda_12_slot_corner_radius_ratio: float = field(
        metadata=_lambda_metadata(
            "lambda_12",
            "r_S / (b_S / 2)",
        ),
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

        self._validate_geometry()

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

    def _validate_geometry(self) -> None:
        strictly_positive: dict[str, float] = {
            "lambda_3": self.lambda_3_airgap_magnet_length_ratio,
            "lambda_4": self.lambda_4_slot_width_pitch_ratio,
            "lambda_5": self.lambda_5_slot_width_depth_ratio,
            "lambda_8": self.lambda_8_stator_back_iron_pitch_ratio,
            "lambda_9": self.lambda_9_magnet_rotor_length_ratio,
        }

        for name, value in strictly_positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be strictly positive.")

        if not 0 < self.lambda_1_diameter_ratio < 1:
            raise ValueError("lambda_1 must satisfy 0 < lambda_1 < 1.")

        if not 0 < self.lambda_2_magnet_arc_ratio <= 1:
            raise ValueError("lambda_2 must satisfy 0 < lambda_2 <= 1.")

        if not 0 < self.lambda_6_magnet_radial_ratio < 1:
            raise ValueError("lambda_6 must satisfy 0 < lambda_6 < 1.")

        if not 0 <= self.lambda_7_embed_remaining_width_ratio < 1:
            raise ValueError("lambda_7 must satisfy 0 <= lambda_7 < 1.")

        bounded_ratios: dict[str, float] = {
            "lambda_10": self.lambda_10_slot_opening_width_ratio,
            "lambda_11": self.lambda_11_tooth_tip_depth_ratio,
            "lambda_12": self.lambda_12_slot_corner_radius_ratio,
        }

        for name, value in bounded_ratios.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must satisfy 0 <= {name} <= 1.")


def geometry_from_lambdas(
    lambdas: LambdaValues,
    *,
    airgap: float,
    stator_bore: float,
    slot_number: int,
    pole_number: int,
) -> dict[str, float]:
    """Calculate geometry dimensions from lambda values and base dimensions."""
    if not isinstance(lambdas, LambdaValues):
        raise TypeError("lambdas must be a LambdaValues instance.")

    for name, value in (
        ("airgap", airgap),
        ("stator_bore", stator_bore),
    ):
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be a real number.")

        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and strictly positive.")

    for name, count in (
        ("slot_number", slot_number),
        ("pole_number", pole_number),
    ):
        if type(count) is not int or count < 1:
            raise ValueError(f"{name} must be a positive integer.")

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
    # w*d_a/2 = 2*pi*n*d_a/2 = v_umfang [mm / min]

    g: float = float(airgap)
    d_si: float = float(stator_bore)

    d_sa = d_si / lambdas.lambda_1_diameter_ratio
    beta_pm = lambdas.lambda_2_magnet_arc_ratio * 180
    l_pm = g / lambdas.lambda_3_airgap_magnet_length_ratio

    b_s = lambdas.lambda_4_slot_width_pitch_ratio * math.pi * d_si / slot_number
    d_s = b_s / lambdas.lambda_5_slot_width_depth_ratio

    radial_width = (d_sa - d_si) / 2
    h_pm = lambdas.lambda_6_magnet_radial_ratio * radial_width
    d_emb = lambdas.lambda_7_embed_remaining_width_ratio * (radial_width - h_pm)

    d_mean = (d_sa + d_si) / 2
    l_s = (
        d_s
        + lambdas.lambda_8_stator_back_iron_pitch_ratio * math.pi * d_mean / pole_number
    )
    l_r = l_pm / lambdas.lambda_9_magnet_rotor_length_ratio

    b_tto = lambdas.lambda_10_slot_opening_width_ratio * b_s
    d_ttd = lambdas.lambda_11_tooth_tip_depth_ratio * d_s
    r_s = lambdas.lambda_12_slot_corner_radius_ratio * (b_s / 2)

    geometry: dict[str, float] = {
        "Slot_Number": float(slot_number),
        "Pole_Number": float(pole_number),
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
    }

    if not all(math.isfinite(value) for value in geometry.values()):
        raise ValueError("Geometry calculation produced non-finite values.")

    return geometry


def lambdas_from_geometry(
    geometry: Mapping[str, float],
) -> LambdaValues:
    missing = set(GEOMETRY_INPUT_COLUMNS) - set(geometry)
    if missing:
        raise ValueError(f"Missing geometry parameters: {sorted(missing)}")

    values: dict[str, float] = {}
    for name in GEOMETRY_INPUT_COLUMNS:
        value = geometry[name]

        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be a real number.")

        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")

        values[name] = float(value)

    strictly_positive: tuple[str, ...] = (
        "Slot_Number",
        "Pole_Number",
        "Airgap",
        "Stator_Bore",
        "Stator_Lam_Dia",
        "Magnet_Length",
        "Slot_Width",
        "Slot_Depth",
        "Rotor_Lam_Length",
    )

    for name in strictly_positive:
        if values[name] <= 0:
            raise ValueError(f"{name} must be strictly positive.")

    for name in ("Slot_Number", "Pole_Number"):
        if not values[name].is_integer():
            raise ValueError(f"{name} must be an integer.")

    d_si = values["Stator_Bore"]
    d_sa = values["Stator_Lam_Dia"]
    radial_width = (d_sa - d_si) / 2
    h_pm = values["Magnet_Thickness"]

    if radial_width <= 0:
        raise ValueError("Radial width must be strictly positive.")

    remaining_width = radial_width - h_pm
    if remaining_width <= 0:
        raise ValueError("Radial width minus magnet thickness must be positive.")

    b_s = values["Slot_Width"]
    d_s = values["Slot_Depth"]

    return LambdaValues(
        lambda_1_diameter_ratio=d_si / d_sa,
        lambda_2_magnet_arc_ratio=values["Magnet_Arc_[ED]"] / 180,
        lambda_3_airgap_magnet_length_ratio=(
            values["Airgap"] / values["Magnet_Length"]
        ),
        lambda_4_slot_width_pitch_ratio=(
            values["Slot_Number"] * b_s / (math.pi * d_si)
        ),
        lambda_5_slot_width_depth_ratio=b_s / d_s,
        lambda_6_magnet_radial_ratio=h_pm / radial_width,
        lambda_7_embed_remaining_width_ratio=(
            values["Magnet_Embed_Depth"] / remaining_width
        ),
        lambda_8_stator_back_iron_pitch_ratio=(
            2
            * values["Pole_Number"]
            * (values["Stator_Lam_Length"] - d_s)
            / (math.pi * (d_sa + d_si))
        ),
        lambda_9_magnet_rotor_length_ratio=(
            values["Magnet_Length"] / values["Rotor_Lam_Length"]
        ),
        lambda_10_slot_opening_width_ratio=values["Slot_Opening"] / b_s,
        lambda_11_tooth_tip_depth_ratio=values["Tooth_Tip_Depth"] / d_s,
        lambda_12_slot_corner_radius_ratio=(values["Slot_Corner_Radius"] / (b_s / 2)),
    )
