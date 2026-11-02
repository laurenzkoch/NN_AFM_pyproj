from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config.config_mcad_parameters import AXIS_LIST


class GridAligner:
    def __init__(
        self,
        map_axes: dict[str, NDArray[np.float64]],
        csv_decimal_places: int,
    ) -> None:
        self.map_axes = map_axes
        self.csv_decimal_places = csv_decimal_places

    def align_axis_grid(
        self,
        df: pd.DataFrame,
        axis_name: str,
        possible_values: NDArray[np.float64],
        positions: NDArray[np.intp],
    ) -> None:
        """Assign exact grid values where rounding allows one unique match."""
        numerical_tolerance: float = 1e-10

        # First dimension: CSV rows; remaining dimensions: rounding cases.
        bounds: NDArray[np.float64] = possible_values.reshape(
            positions.size,
            -1,
        )
        lower = bounds.min(axis=1)
        upper = bounds.max(axis=1)

        axis: NDArray[np.float64] = self.map_axes[axis_name]
        candidates = (axis[None, :] >= lower[:, None] - numerical_tolerance) & (
            axis[None, :] <= upper[:, None] + numerical_tolerance
        )
        unique = candidates.sum(axis=1) == 1

        aligned: NDArray[np.float64] = df[axis_name].to_numpy(
            dtype=np.float64,
            copy=True,
        )
        aligned[positions[unique]] = axis[candidates[unique].argmax(axis=1)]
        df[axis_name] = aligned

    def align_speed_grid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recover speed-grid coordinates within CSV rounding bounds."""
        if df.empty:
            return df

        half_resolution = 0.5 * 10.0 ** (-self.csv_decimal_places)
        offsets = np.array(
            [-half_resolution, half_resolution],
            dtype=np.float64,
        )

        speed = df["ShaftSpeed"].to_numpy(dtype=np.float64)
        base_speed = df["BaseSpeedRPM"].to_numpy(dtype=np.float64)

        if bool((base_speed <= half_resolution).any()):
            raise ValueError("Base speed is too small for speed-grid alignment.")

        speed_bounds = speed[:, None] + offsets
        base_speed_bounds = base_speed[:, None] + offsets

        possible_values = speed_bounds[:, :, None] / base_speed_bounds[:, None, :]

        self.align_axis_grid(
            df=df,
            axis_name=AXIS_LIST[0],
            possible_values=possible_values,
            positions=np.arange(len(df), dtype=np.intp),
        )

        return df

    def align_current_density_grid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recover current-density coordinates within CSV rounding bounds."""
        half_resolution = 0.5 * 10.0 ** (-self.csv_decimal_places)
        offsets: NDArray[np.float64] = np.array(
            [-half_resolution, half_resolution],
            dtype=np.float64,
        )

        peak_current = df["PeakCurrent"].to_numpy(dtype=np.float64)
        density = df["RMSCurrentDensity"].to_numpy(dtype=np.float64)

        usable = (peak_current > half_resolution) & (density > half_resolution)
        positions: NDArray[np.intp] = np.flatnonzero(usable)

        if positions.size == 0:
            return df

        peak_bounds = peak_current[usable, None] + offsets
        density_bounds = density[usable, None] + offsets

        for axis_name, current_name in (
            (AXIS_LIST[1], "RMSPhaseCurrent_Q"),
            (AXIS_LIST[2], "RMSPhaseCurrent_D"),
        ):
            current = df[current_name].to_numpy(dtype=np.float64)
            current_bounds = current[usable, None] + offsets

            possible_values = (
                np.sqrt(2.0)
                * current_bounds[:, :, None, None]
                * density_bounds[:, None, :, None]
                / peak_bounds[:, None, None, :]
            )

            self.align_axis_grid(
                df=df,
                axis_name=axis_name,
                possible_values=possible_values,
                positions=positions,
            )

        return df
