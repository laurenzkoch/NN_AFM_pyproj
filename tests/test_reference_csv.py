from pathlib import Path

import numpy as np
import pytest

from src.config.config_mcad_parameters import (
    AXIS_LIST,
    CSV_PARAMETERS,
)
from src.modules.data_managment.csv_io import (
    create_csv_file,
    import_data_from_csv,
    write_row_in_csv,
)
from src.modules.neuronal_networks.manipulating_data import DataLoader
from tests.conftest import make_frame


def test_csv_roundtrip_reconstructs_normalized_axes(
    tmp_path: Path,
) -> None:
    original = make_frame(machine_count=2)
    csv_path = tmp_path / "reference_roundtrip.csv"

    create_csv_file(csv_path, CSV_PARAMETERS)

    for row in original.itertuples(index=False, name=None):
        write_row_in_csv(csv_path, row)

    imported = import_data_from_csv(csv_path)
    reconstructed = DataLoader._recalculate_data(imported)

    assert reconstructed.columns.is_unique
    assert len(reconstructed) == len(original)

    np.testing.assert_allclose(
        reconstructed[AXIS_LIST[0]],
        original["ShaftSpeedNorm"],
        rtol=0.0,
        atol=1e-10,
    )

    # The synthetic test data uses a conductor area of 1 mm².
    np.testing.assert_allclose(
        reconstructed[AXIS_LIST[1]],
        original["RMSPhaseCurrent_Q"],
        rtol=0.0,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        reconstructed[AXIS_LIST[2]],
        original["RMSPhaseCurrent_D"],
        rtol=0.0,
        atol=1e-10,
    )


def test_each_machine_uses_its_own_base_speed() -> None:
    frame = make_frame(machine_count=2)
    reconstructed = DataLoader._recalculate_data(frame)

    first = reconstructed[reconstructed["Stator_Bore"] == 70.0]
    second = reconstructed[reconstructed["Stator_Bore"] == 71.0]

    assert first["BaseSpeedRPM"].nunique() == 1
    assert second["BaseSpeedRPM"].nunique() == 1

    assert first["BaseSpeedRPM"].iloc[0] != second["BaseSpeedRPM"].iloc[0]

    np.testing.assert_allclose(
        first[AXIS_LIST[0]].to_numpy(),
        second[AXIS_LIST[0]].to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )

    assert not np.allclose(
        first["ShaftSpeed"].to_numpy(),
        second["ShaftSpeed"].to_numpy(),
    )


@pytest.mark.parametrize(
    "column",
    [
        "BaseSpeedRPM",
        "ReferenceKeFundamental",
        "ShaftSpeedNorm",
    ],
)
def test_inconsistent_reference_data_is_rejected(
    column: str,
) -> None:
    frame = make_frame(machine_count=1)
    frame.loc[0, column] *= 2.0

    with pytest.raises(ValueError):
        DataLoader._recalculate_data(frame)


@pytest.mark.parametrize(
    "column",
    [
        "BaseSpeedRPM",
        "ReferenceKeFundamental",
        "ReferencePeakBackEMFPhase",
        "ReferenceSpeedRPM",
        "DCBusVoltage",
    ],
)
@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf")],
)
def test_nonfinite_reference_data_is_rejected(
    column: str,
    value: float,
) -> None:
    frame = make_frame(machine_count=1)
    frame.loc[0, column] = value

    with pytest.raises(ValueError):
        DataLoader._recalculate_data(frame)
