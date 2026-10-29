from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from src.config.config_mcad_parameters import (
    CSV_DECIMAL_PLACES,
    CSV_PARAMETERS,
    REFERENCE_SPEED_RPM,
)
from src.modules.data_managment import simulation_metadata as module
from src.modules.data_managment.experiment_utils import canonical_hash, file_sha256
from src.modules.data_managment.simulation_metadata import SimulationMetadata


def legacy_csv(path: Path) -> None:
    pd.DataFrame([[1.0] * len(CSV_PARAMETERS)], columns=CSV_PARAMETERS).to_csv(
        path, sep=";", decimal=",", index=False
    )


def adopt(
    path: Path, metadata: SimulationMetadata, *, write: bool = False
) -> SimulationMetadata:
    return module.adopt_legacy_csv(
        path,
        metadata,
        confirmed_mot_hash=metadata["payload"]["base_model_sha256"],
        reason="Synthetischer Regressionstest",
        accept_setup_assumption=True,
        write=write,
    )


def test_adoption_dry_run_does_not_write(
    tmp_path: Path, metadata: SimulationMetadata
) -> None:
    path = tmp_path / "legacy.csv"
    legacy_csv(path)
    before = file_sha256(path)
    result = adopt(path, metadata)
    assert result["adoption"]["original_setup_verified"] is False
    assert file_sha256(path) == before
    assert not module.metadata_path(path).exists()


def test_adoption_roundtrip_and_tamper_detection(
    tmp_path: Path, metadata: SimulationMetadata
) -> None:
    path = tmp_path / "legacy.csv"
    legacy_csv(path)
    before = file_sha256(path)
    document = adopt(path, metadata, write=True)
    assert module.read_metadata(path) == document
    assert file_sha256(path) == before
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="subsequently"):
        module.read_metadata(path)


@pytest.mark.parametrize("problem", ["empty", "columns", "nonfinite"])
def test_bad_legacy_sources_rejected(
    tmp_path: Path, metadata: SimulationMetadata, problem: str
) -> None:
    path = tmp_path / "bad.csv"
    frame = pd.DataFrame([[1.0] * len(CSV_PARAMETERS)], columns=CSV_PARAMETERS)
    if problem == "empty":
        frame = frame.iloc[:0]
    elif problem == "columns":
        frame = frame[CSV_PARAMETERS[::-1]]
    else:
        frame.iloc[0, 0] = float("inf")
    frame.to_csv(path, sep=";", decimal=",", index=False)
    with pytest.raises(ValueError):
        adopt(path, metadata, write=True)
    assert not module.metadata_path(path).exists()


@pytest.mark.parametrize(
    "confirmed,accepted,reason",
    [("0" * 64, True, "test"), ("1" * 64, False, "test"), ("1" * 64, True, " ")],
)
def test_adoption_requires_explicit_valid_confirmation(
    tmp_path: Path,
    metadata: SimulationMetadata,
    confirmed: str,
    accepted: bool,
    reason: str,
) -> None:
    path = tmp_path / "legacy.csv"
    legacy_csv(path)
    with pytest.raises(ValueError):
        module.adopt_legacy_csv(
            path,
            metadata,
            confirmed_mot_hash=confirmed,
            reason=reason,
            accept_setup_assumption=accepted,
            write=True,
        )
    assert not module.metadata_path(path).exists()


def test_native_output_and_adopted_output_are_distinct(
    tmp_path: Path, metadata: SimulationMetadata
) -> None:
    native = tmp_path / "native.csv"
    module.prepare_result_csv(native, metadata)
    assert module.validate_result_csv(native).empty
    assert module.read_metadata(native) == metadata
    module.prepare_result_csv(native, metadata)
    legacy = tmp_path / "legacy.csv"
    legacy_csv(legacy)
    adopt(legacy, metadata, write=True)
    with pytest.raises(ValueError, match="cache source"):
        module.prepare_result_csv(legacy, metadata)
    with pytest.raises(FileExistsError):
        adopt(legacy, metadata, write=True)


def test_corrupt_fingerprint_and_orphan_sidecar_rejected(
    tmp_path: Path, metadata: SimulationMetadata
) -> None:
    path = tmp_path / "native.csv"
    broken: SimulationMetadata = deepcopy(metadata)
    broken["fingerprint"] = "0" * 64
    module.write_metadata(path, broken)
    with pytest.raises(ValueError, match="fingerprint"):
        module.read_metadata(path)
    with pytest.raises(FileExistsError):
        module.prepare_result_csv(path, metadata)


def test_version_3_requires_matching_setup_and_valid_fingerprints(
    metadata: SimulationMetadata,
) -> None:
    assert module.is_compatible_metadata(metadata, metadata)
    assert "implementation_hashes" not in metadata["payload"]

    changed_model: SimulationMetadata = deepcopy(metadata)
    changed_model["payload"]["base_model_sha256"] = "4" * 64
    changed_model["fingerprint"] = canonical_hash(changed_model["payload"])

    assert not module.is_compatible_metadata(metadata, changed_model)

    invalid: SimulationMetadata = deepcopy(metadata)
    invalid["fingerprint"] = "0" * 64

    assert not module.is_compatible_metadata(invalid, metadata)
    assert not module.is_compatible_metadata(invalid, invalid)

    old_version: SimulationMetadata = deepcopy(metadata)
    old_version["payload"]["metadata_version"] = 1
    old_version["fingerprint"] = canonical_hash(old_version["payload"])

    assert not module.is_compatible_metadata(old_version, metadata)


def test_metadata_builder_does_not_hash_python_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hashed_paths: list[Path] = []

    def fake_file_hash(path: str | Path) -> str:
        resolved_path: Path = Path(path)

        if resolved_path.suffix.lower() == ".py":
            raise AssertionError("Python files must not affect simulation metadata.")

        hashed_paths.append(resolved_path)
        return "1" * 64

    def fake_version(package_name: str) -> str:
        assert package_name == "ansys-motorcad-core"
        return "synthetic-test"

    monkeypatch.setattr(
        "src.modules.data_managment.simulation_metadata.file_sha256",
        fake_file_hash,
    )
    monkeypatch.setattr(
        "src.modules.data_managment.simulation_metadata.version",
        fake_version,
    )
    monkeypatch.setattr(
        "src.modules.data_managment.simulation_metadata.MCAD_DEPENDENCY_FILES",
        (),
    )

    result: SimulationMetadata = module.build_simulation_metadata()

    assert result["payload"]["metadata_version"] == 3
    assert result["payload"]["float_decimal_places"] == CSV_DECIMAL_PLACES
    assert result["payload"]["reference_speed_rpm"] == REFERENCE_SPEED_RPM
    assert result["payload"]["reference_current_a"] == 0.0

    assert result["payload"]["speed_normalization"] == (
        "shaft_speed_rpm / base_speed_rpm"
    )

    assert result["payload"]["back_emf_constant_definition"] == (
        "fundamental line-line peak / mechanical angular speed; V*s/rad"
    )
    assert "implementation_hashes" not in result["payload"]
    assert hashed_paths == [
        Path(module.PATH_MCAD_BASEFILE),
        Path(module.PATH_MCAD_EXE),
    ]
    assert result["fingerprint"] == canonical_hash(result["payload"])


@pytest.mark.parametrize("previous_version", [1, 2])
def test_old_metadata_versions_are_incompatible(
    metadata: SimulationMetadata,
    previous_version: int,
) -> None:
    previous: SimulationMetadata = deepcopy(metadata)
    previous["payload"]["metadata_version"] = previous_version
    previous["fingerprint"] = canonical_hash(previous["payload"])

    assert not module.is_compatible_metadata(previous, metadata)
    assert not module.is_compatible_metadata(metadata, previous)


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("reference_speed_rpm", 2000.0),
        ("reference_current_a", 1.0),
        ("speed_normalization", "different normalization"),
        ("back_emf_constant_definition", "different definition"),
        ("float_decimal_places", 4),
    ],
)
def test_changed_reference_setup_is_incompatible(
    metadata: SimulationMetadata,
    field: str,
    changed_value: object,
) -> None:
    changed: SimulationMetadata = deepcopy(metadata)

    # Dynamic field names are intentional in this parametrized test.
    payload: dict[str, object] = dict(changed["payload"])
    payload[field] = changed_value

    changed["payload"] = cast(module.SimulationPayload, payload)
    changed["fingerprint"] = canonical_hash(changed["payload"])

    assert not module.is_compatible_metadata(changed, metadata)
