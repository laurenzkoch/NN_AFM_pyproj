import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.modules.data_managment.experiment_utils import (
    canonical_hash,
)
from src.modules.data_managment.simulation_metadata import (
    SimulationMetadata,
    metadata_path,
    migrate_simulation_metadata,
    read_metadata,
)
from tests.conftest import make_frame, write_frame


@pytest.mark.parametrize(
    "previous_version",
    [1, 2, 3, 4],
)
def test_general_migration_preserves_csv(
    tmp_path: Path,
    metadata: SimulationMetadata,
    previous_version: int,
) -> None:
    path = tmp_path / "source.csv"
    write_frame(path, make_frame())

    sidecar = metadata_path(path)

    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    raw["payload"]["metadata_version"] = previous_version

    if previous_version < 4:
        raw["payload"].pop("map_grid")

    if previous_version == 1:
        raw["payload"]["implementation_hashes"] = {
            "old.py": "old-hash",
        }

    raw["fingerprint"] = canonical_hash(raw["payload"])

    sidecar.write_text(
        json.dumps(raw),
        encoding="utf-8",
    )

    before_csv = path.read_bytes()
    before_sidecar = sidecar.read_bytes()

    migrated = migrate_simulation_metadata(
        path,
        metadata,
    )

    assert migrated == metadata
    assert sidecar.read_bytes() == before_sidecar
    assert path.read_bytes() == before_csv

    migrate_simulation_metadata(
        path,
        metadata,
        write=True,
    )

    assert read_metadata(path) == metadata
    assert path.read_bytes() == before_csv


def test_general_migration_refuses_other_model_setup(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.csv"
    write_frame(path, make_frame())

    expected: SimulationMetadata = deepcopy(read_metadata(path))

    expected["payload"]["base_model_sha256"] = "9" * 64
    expected["fingerprint"] = canonical_hash(expected["payload"])

    before_csv = path.read_bytes()
    before_sidecar = metadata_path(path).read_bytes()

    with pytest.raises(
        ValueError,
        match="setup differs",
    ):
        migrate_simulation_metadata(
            path,
            expected,
            write=True,
        )

    assert metadata_path(path).read_bytes() == before_sidecar
    assert path.read_bytes() == before_csv
