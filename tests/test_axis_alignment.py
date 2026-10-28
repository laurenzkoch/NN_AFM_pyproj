from pathlib import Path

import numpy as np
import pandas as pd

from src.config.config_mcad_parameters import (
    AXIS_LIST,
    CSV_DECIMAL_PLACES,
)
from src.modules.neuronal_networks.manipulating_data import DataLoader


def rounded(value: float) -> float:
    return float(f"{value:.{CSV_DECIMAL_PLACES}f}")


def test_speed_grid_corrects_csv_rounding(tmp_path: Path) -> None:
    loader = DataLoader(tmp_path)
    axis_name = AXIS_LIST[0]

    loader.map_axes[axis_name] = np.array([0.0, 10.0, 20.0])

    base_speed = 123.4567890123456
    shaft_speed = 10.0 * base_speed

    df = pd.DataFrame(
        {
            "ShaftSpeed": [rounded(shaft_speed)],
            "BaseSpeedRPM": [rounded(base_speed)],
        },
        index=[42],
    )

    df[axis_name] = df["ShaftSpeed"] / df["BaseSpeedRPM"]

    stored_speed = df["ShaftSpeed"].iloc[0]
    stored_base_speed = df["BaseSpeedRPM"].iloc[0]

    result = loader._align_speed_grid(df)

    assert result[axis_name].iloc[0] == 10.0
    assert result["ShaftSpeed"].iloc[0] == stored_speed
    assert result["BaseSpeedRPM"].iloc[0] == stored_base_speed
    assert result.index.tolist() == [42]


def test_speed_grid_preserves_off_grid_value(tmp_path: Path) -> None:
    loader = DataLoader(tmp_path)
    axis_name = AXIS_LIST[0]

    loader.map_axes[axis_name] = np.array([0.0, 10.0, 20.0])

    df = pd.DataFrame(
        {
            "ShaftSpeed": [1100.0],
            "BaseSpeedRPM": [100.0],
        }
    )

    df[axis_name] = df["ShaftSpeed"] / df["BaseSpeedRPM"]

    result = loader._align_speed_grid(df)

    assert result[axis_name].iloc[0] == 11.0


def test_axis_grid_preserves_ambiguous_value(tmp_path: Path) -> None:
    loader = DataLoader(tmp_path)
    axis_name = AXIS_LIST[0]

    loader.map_axes[axis_name] = np.array([10.0, 20.0])

    df = pd.DataFrame({axis_name: [15.0]}, index=[42])

    loader._align_axis_grid(
        df=df,
        axis_name=axis_name,
        possible_values=np.array([[9.0, 21.0]]),
        positions=np.array([0], dtype=np.intp),
    )

    assert df[axis_name].iloc[0] == 15.0


def test_current_density_grid_corrects_csv_rounding(
    tmp_path: Path,
) -> None:
    loader = DataLoader(tmp_path)

    loader.map_axes[AXIS_LIST[1]] = np.array([0.0, 5.0, 10.0])
    loader.map_axes[AXIS_LIST[2]] = np.array([-5.0, 0.0, 5.0])

    df = pd.DataFrame(
        {
            "PeakCurrent": [rounded(np.sqrt(2.0) * 10.0)],
            "RMSCurrentDensity": [10.0],
            "RMSPhaseCurrent_D": [-5.0],
            "RMSPhaseCurrent_Q": [10.0],
        },
        index=[42],
    )

    conductor_area = df["PeakCurrent"] / (
        np.sqrt(2.0) * df["RMSCurrentDensity"]
    )

    df[AXIS_LIST[1]] = (
        df["RMSPhaseCurrent_Q"] / conductor_area
    )
    df[AXIS_LIST[2]] = (
        df["RMSPhaseCurrent_D"] / conductor_area
    )

    result = loader._align_current_density_grid(df)

    assert result[AXIS_LIST[1]].iloc[0] == 10.0
    assert result[AXIS_LIST[2]].iloc[0] == -5.0
    assert result.index.tolist() == [42]