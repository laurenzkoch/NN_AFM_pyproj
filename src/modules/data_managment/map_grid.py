"""Describe the operating grid used by simulation and training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config.config_mcad_parameters import (
    AXIS_LIST,
    D_CURRENT_DENSITY,
    MAP_SIZE,
    Q_CURRENT_DENSITY,
    SHAFT_SPEED_NORM,
)


class MapGridPayload(TypedDict):
    shaft_speed_norm: list[float]
    q_current_density: list[float]
    d_current_density: list[float]
    map_size: list[int]


@dataclass(frozen=True)
class MapGrid:
    shaft_speed_norm: tuple[float, float]
    q_current_density: tuple[float, float]
    d_current_density: tuple[float, float]
    map_size: tuple[int, int, int]

    def __post_init__(self) -> None:
        bounds = (
            self.shaft_speed_norm,
            self.q_current_density,
            self.d_current_density,
        )

        if len(self.map_size) != 3 or any(
            type(size) is not int or size < 1 for size in self.map_size
        ):
            raise ValueError("map_size must contain three positive integers.")

        for axis, size in zip(bounds, self.map_size, strict=True):
            if len(axis) != 2 or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in axis
            ):
                raise ValueError("Each map-grid range needs two finite numbers.")

            if axis[0] > axis[1] or (size > 1 and axis[0] == axis[1]):
                raise ValueError("Map-grid bounds must increase for multiple points.")

        if self.shaft_speed_norm[0] < 0:
            raise ValueError("Normalized speed must not be negative.")

    @classmethod
    def from_config(cls) -> MapGrid:
        return cls(
            shaft_speed_norm=SHAFT_SPEED_NORM,
            q_current_density=Q_CURRENT_DENSITY,
            d_current_density=D_CURRENT_DENSITY,
            map_size=MAP_SIZE,
        )

    def to_payload(self) -> MapGridPayload:
        return {
            "shaft_speed_norm": [float(value) for value in self.shaft_speed_norm],
            "q_current_density": [float(value) for value in self.q_current_density],
            "d_current_density": [float(value) for value in self.d_current_density],
            "map_size": list(self.map_size),
        }

    @classmethod
    def from_payload(cls, raw: object) -> MapGrid:
        """Restore and validate an operating grid from JSON metadata."""
        expected_keys = {
            "shaft_speed_norm",
            "q_current_density",
            "d_current_density",
            "map_size",
        }

        if not isinstance(raw, dict) or set(raw) != expected_keys:
            raise ValueError("Metadata has no valid map_grid; migrate it explicitly.")

        ranges: list[tuple[float, float]] = []

        for name in (
            "shaft_speed_norm",
            "q_current_density",
            "d_current_density",
        ):
            values = raw[name]

            if (
                not isinstance(values, list)
                or len(values) != 2
                or any(type(value) not in (int, float) for value in values)
            ):
                raise ValueError(f"Invalid map-grid range: {name}")

            ranges.append((float(values[0]), float(values[1])))

        sizes = raw["map_size"]

        if not isinstance(sizes, list) or len(sizes) != 3:
            raise ValueError("map_size must contain three positive integers.")

        return cls(
            shaft_speed_norm=ranges[0],
            q_current_density=ranges[1],
            d_current_density=ranges[2],
            map_size=(sizes[0], sizes[1], sizes[2]),
        )

    def axes(self) -> dict[str, NDArray[np.float64]]:
        return {
            name: np.linspace(
                *bounds,
                size,
                dtype=np.float64,
            )
            for name, bounds, size in zip(
                AXIS_LIST,
                (
                    self.shaft_speed_norm,
                    self.q_current_density,
                    self.d_current_density,
                ),
                self.map_size,
                strict=True,
            )
        }

    def validate_frame(self, frame: pd.DataFrame) -> None:
        """Check the declared grid against observed CSV coordinates."""
        if frame.empty:
            raise ValueError("Cannot confirm a map grid from an empty CSV.")

        conductor_area = frame["PeakCurrent"] / (
            np.sqrt(2.0) * frame["RMSCurrentDensity"]
        )

        coordinates = (
            frame["ShaftSpeed"] / frame["BaseSpeedRPM"],
            frame["RMSPhaseCurrent_Q"] / conductor_area,
            frame["RMSPhaseCurrent_D"] / conductor_area,
        )

        if not np.allclose(
            coordinates[0],
            frame["ShaftSpeedNorm"],
            rtol=1e-8,
            atol=1e-9,
        ):
            raise ValueError("Stored and reconstructed normalized speeds disagree.")

        for (name, axis), values in zip(
            self.axes().items(),
            coordinates,
            strict=True,
        ):
            matches = np.isclose(
                values.to_numpy(dtype=float)[:, None],
                axis[None, :],
                rtol=1e-8,
                atol=1e-9,
            )

            unique_match_per_row = (matches.sum(axis=1) == 1).all()

            every_axis_value_observed = (matches.any(axis=0)).all()

            if not unique_match_per_row or not every_axis_value_observed:
                raise ValueError(f"CSV does not confirm the declared grid: {name}")
