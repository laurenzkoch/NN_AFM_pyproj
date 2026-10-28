import csv
import json
import os
import tempfile
from copy import deepcopy
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

import numpy as np
import pandas as pd

from src.config.config_mcad_parameters import (
    CSV_DECIMAL_PLACES,
    CSV_PARAMETERS,
    REFERENCE_SPEED_RPM,
)
from src.config.config_paths import (
    MCAD_DEPENDENCY_FILES,
    MCAD_DEPENDENCY_REVISION,
    PATH_MCAD_BASEFILE,
    PATH_MCAD_EXE,
)
from src.modules.data_managment.csv_io import import_data_from_csv
from src.modules.data_managment.experiment_utils import canonical_hash, file_sha256


class SimulationPayload(TypedDict):
    metadata_version: int
    base_model_sha256: str
    mcad_executable_sha256: str
    integration_version: str
    external_dependency_revision: str
    external_dependency_hashes: dict[str, str]
    csv_columns: list[str]
    float_decimal_places: int
    speed_normalization: str
    reference_speed_rpm: float
    reference_current_a: float
    back_emf_constant_definition: str


class LegacyAdoption(TypedDict):
    kind: Literal["legacy_assumption"]
    reason: str
    accepted_at_utc: str
    csv_sha256: str
    original_setup_verified: Literal[False]


class SimulationMetadata(TypedDict):
    fingerprint: str
    payload: SimulationPayload
    adoption: NotRequired[LegacyAdoption]


def _copy_simulation_payload(
    payload: SimulationPayload,
) -> SimulationPayload:
    return deepcopy(payload)

def is_compatible_metadata(
    actual: SimulationMetadata,
    expected: SimulationMetadata,
) -> bool:
    if (
        actual["payload"].get("metadata_version") != 3
        or expected["payload"].get("metadata_version") != 3
    ):
        return False

    if actual["payload"] != expected["payload"]:
        return False

    calculated_fingerprint: str = canonical_hash(expected["payload"])

    return actual["fingerprint"] == expected["fingerprint"] == calculated_fingerprint


def metadata_path(csv_path: str | Path) -> Path:
    path = Path(csv_path)
    return path.with_suffix(path.suffix + ".meta.json")


def build_simulation_metadata() -> SimulationMetadata:
    payload: SimulationPayload = {
        "metadata_version": 3,
        "base_model_sha256": file_sha256(PATH_MCAD_BASEFILE),
        "mcad_executable_sha256": file_sha256(PATH_MCAD_EXE),
        "integration_version": version("ansys-motorcad-core"),
        "external_dependency_revision": MCAD_DEPENDENCY_REVISION,
        "external_dependency_hashes": {
            str(Path(path).resolve()): file_sha256(path)
            for path in MCAD_DEPENDENCY_FILES
        },
        "csv_columns": list(CSV_PARAMETERS),
        "float_decimal_places": CSV_DECIMAL_PLACES,
        "speed_normalization": "shaft_speed_rpm / base_speed_rpm",
        "reference_speed_rpm": REFERENCE_SPEED_RPM,
        "reference_current_a": 0.0,
        "back_emf_constant_definition": (
            "fundamental line-line peak / mechanical angular speed; V*s/rad"
        ),
    }

    return {
        "fingerprint": canonical_hash(payload),
        "payload": payload,
    }


def validate_result_csv(csv_path: Path) -> pd.DataFrame:
    with csv_path.open(encoding="utf-8", newline="") as stream:
        header = next(
            csv.reader(stream, delimiter=";"),
            None,
        )

    if header != CSV_PARAMETERS:
        raise ValueError(
            f"The CSV columns or their order are incorrect: {csv_path.name}"
        )

    frame = import_data_from_csv(csv_path)
    values = frame[CSV_PARAMETERS].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"The CSV file contains NaN or infinite values: {csv_path.name}"
        )

    return frame.reset_index(drop=True)


def read_metadata(csv_path: Path) -> SimulationMetadata:
    raw: object = json.loads(metadata_path(csv_path).read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError("Metadata must be a JSON object.")

    if not isinstance(raw.get("payload"), dict):
        raise ValueError("The metadata payload is missing or invalid.")

    if not isinstance(raw.get("fingerprint"), str):
        raise ValueError("The simulated fingerprint is missing.")

    document = cast(SimulationMetadata, raw)

    if document["fingerprint"] != canonical_hash(document["payload"]):
        raise ValueError("The metadata fingerprint is incorrect.")

    if "adoption" in raw:
        adoption = raw["adoption"]

        if not isinstance(adoption, dict):
            raise ValueError("Invalid acceptance note.")

        if (
            adoption.get("kind") != "legacy_assumption"
            or adoption.get("original_setup_verified") is not False
        ):
            raise ValueError("Invalid label for the old file.")

        for key in ("reason", "accepted_at_utc", "csv_sha256"):
            value = adoption.get(key)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Invalid transfer field: {key}")

        if adoption["csv_sha256"] != file_sha256(csv_path):
            raise ValueError("The imported CSV file was subsequently amended.")

    return document


def write_metadata(
    csv_path: Path,
    document: SimulationMetadata,
) -> None:
    destination: Path = metadata_path(csv_path)

    stream = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix="metadata_",
        suffix=".tmp",
        delete=False,
    )
    temporary: Path = Path(stream.name)

    try:
        with stream:
            json.dump(
                document,
                stream,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )

        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_result_csv(
    csv_path: Path,
    expected: SimulationMetadata,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        actual = read_metadata(csv_path)

        if actual["fingerprint"] != expected["fingerprint"]:
            raise ValueError("The target file belongs to a different simulation setup.")

        if "adoption" in actual:
            raise ValueError(
                "Imported legacy files can only be used as a cache source. "
                "Use a new destination file."
            )

        validate_result_csv(csv_path)
        return

    if metadata_path(csv_path).exists():
        raise FileExistsError(
            "A metadata file without an associated CSV file already exists."
        )

    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        csv.writer(stream, delimiter=";").writerow(CSV_PARAMETERS)

    try:
        write_metadata(csv_path, expected)
    except BaseException:
        csv_path.unlink(missing_ok=True)
        raise


def adopt_legacy_csv(
    csv_path: Path,
    expected: SimulationMetadata,
    *,
    confirmed_mot_hash: str,
    reason: str,
    accept_setup_assumption: bool,
    write: bool = False,
) -> SimulationMetadata:
    if not accept_setup_assumption or not reason.strip():
        raise ValueError(
            "The acceptance of the structure must be confirmed, stating the reasons."
        )

    if confirmed_mot_hash.strip().lower() != expected["payload"]["base_model_sha256"]:
        raise ValueError("The confirmed MOT-Hash does not match.")

    if metadata_path(csv_path).exists():
        raise FileExistsError("Metadata already exists for this CSV file.")

    csv_hash = file_sha256(csv_path)
    frame = validate_result_csv(csv_path)

    if frame.empty:
        raise ValueError("The old file does not contain any channel_result rows.")

    if file_sha256(csv_path) != csv_hash:
        raise ValueError("The CSV file was modified during the audit.")

    document: SimulationMetadata = {
        "fingerprint": expected["fingerprint"],
        "payload": _copy_simulation_payload(expected["payload"]),
        "adoption": {
            "kind": "legacy_assumption",
            "reason": reason.strip(),
            "accepted_at_utc": datetime.now(UTC).isoformat(),
            "csv_sha256": csv_hash,
            "original_setup_verified": False,
        },
    }

    if write:
        write_metadata(csv_path, document)

    return document


def migrate_simulation_metadata(
    csv_path: Path,
    expected: SimulationMetadata,
    *,
    write: bool = False,
) -> SimulationMetadata:
    if (
        not is_compatible_metadata(expected, expected)
        or "implementation_hashes" in expected["payload"]
    ):
        raise ValueError("Expected metadata must use the new version 3 format.")

    sidecar: Path = metadata_path(csv_path)
    original_sidecar: bytes = sidecar.read_bytes()
    original_csv_hash: str = file_sha256(csv_path)

    # Prüft den bisherigen Fingerprint und bei adoptierten Dateien
    # zusätzlich den bereits gespeicherten CSV-Hash.
    actual: SimulationMetadata = read_metadata(csv_path)

    # Prüft Spaltenreihenfolge und endliche Ergebniswerte.
    validate_result_csv(csv_path)

    previous_version: object = actual["payload"].get("metadata_version")

    if type(previous_version) is not int or previous_version not in (1, 2, 3):
        raise ValueError("Unsupported metadata version.")

    previous_setup: dict[str, object] = {
        key: value for key, value in actual["payload"].items()
    }

    if previous_version == 1:
        implementation_hashes: object = previous_setup.pop(
            "implementation_hashes",
            None,
        )

        if not isinstance(implementation_hashes, dict):
            raise ValueError("Version 1 implementation hashes are missing.")

        previous_setup["metadata_version"] = 2

    if previous_setup != expected["payload"]:
        raise ValueError(
            "Simulation setup differs from the current setup. "
            "Metadata was not migrated."
        )

    migrated: SimulationMetadata = deepcopy(actual)
    migrated["payload"] = _copy_simulation_payload(expected["payload"])
    migrated["fingerprint"] = expected["fingerprint"]

    if file_sha256(csv_path) != original_csv_hash:
        raise ValueError("CSV changed during migration.")

    if sidecar.read_bytes() != original_sidecar:
        raise ValueError("Metadata changed during migration.")

    if write and migrated != actual:
        write_metadata(csv_path, migrated)

    return migrated
