from copy import deepcopy
from pathlib import Path

import pytest

from src.modules.data_managment.experiment_utils import (
    canonical_hash,
)
from src.modules.data_managment.optimization_sources import (
    discover_test_sources,
)
from src.modules.data_managment.simulation_metadata import (
    SimulationMetadata,
    metadata_path,
    write_metadata,
)
from tests.conftest import make_frame, write_frame


@pytest.mark.parametrize(
    "field",
    [
        "shaft_speed_norm",
        "q_current_density",
        "d_current_density",
        "map_size",
    ],
)
def test_discovery_excludes_different_grid(
    tmp_path: Path,
    metadata: SimulationMetadata,
    field: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    write_frame(
        tmp_path / "main.csv",
        make_frame(),
    )
    write_frame(
        tmp_path / "matching.csv",
        make_frame(1, 20),
    )

    different_path = tmp_path / "different.csv"

    write_frame(
        different_path,
        make_frame(1, 30),
    )

    changed: SimulationMetadata = deepcopy(metadata)
    grid = changed["payload"]["map_grid"]

    if field == "map_size":
        grid["map_size"][0] = 6
    elif field == "shaft_speed_norm":
        grid["shaft_speed_norm"][0] = 0.1
    elif field == "q_current_density":
        grid["q_current_density"][0] = 2.0
    else:
        grid["d_current_density"][0] = -9.0

    changed["fingerprint"] = canonical_hash(changed["payload"])
    write_metadata(different_path, changed)

    selected = discover_test_sources(
        tmp_path,
        "main.csv",
    )

    assert selected == ("matching.csv",)
    assert "different.csv" in caplog.text
    assert "reason=incompatible_metadata" in caplog.text


def test_discovery_excludes_missing_and_corrupt_metadata(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    for name in (
        "main.csv",
        "matching.csv",
        "missing.csv",
        "corrupt.csv",
    ):
        write_frame(
            tmp_path / name,
            make_frame(1),
        )

    metadata_path(tmp_path / "missing.csv").unlink()

    metadata_path(tmp_path / "corrupt.csv").write_text(
        "{}",
        encoding="utf-8",
    )

    selected = discover_test_sources(
        tmp_path,
        "main.csv",
    )

    assert selected == ("matching.csv",)
    assert "missing.csv" in caplog.text
    assert "corrupt.csv" in caplog.text
    assert "reason=invalid_metadata" in caplog.text
