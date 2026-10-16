import csv
import json
import os
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

import numpy as np
import pandas as pd

from src.config.config_mcad_parameters import CSV_PARAMETERS
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
    implementation_hashes: dict[str, str]


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


# Geprüfte Kompatibilität: bisheriger Stand -> bereinigter Stand.
# Simulationsberechnungen und Grenzwerte sind unverändert.
REVIEWED_COMPATIBLE_FINGERPRINTS: frozenset[tuple[str, str]] = frozenset(
    {
        (
            "d5afb8f38a099348f12039cf19c2f595542e3aa1d66c12e88d3421d15909de97",
            "46035f0aa47c3fcb0b0399c8e168af99bff776f3d572081d5f97ebd8abe77a61",
        ),
    }
)


def is_compatible_metadata(
    actual: SimulationMetadata,
    expected: SimulationMetadata,
) -> bool:
    if actual["fingerprint"] == expected["fingerprint"]:
        return True

    fingerprint_pair: tuple[str, str] = (
        actual["fingerprint"],
        expected["fingerprint"],
    )

    if fingerprint_pair not in REVIEWED_COMPATIBLE_FINGERPRINTS:
        return False

    actual_setup: dict[str, object] = {
        key: value
        for key, value in actual["payload"].items()
        if key != "implementation_hashes"
    }
    expected_setup: dict[str, object] = {
        key: value
        for key, value in expected["payload"].items()
        if key != "implementation_hashes"
    }

    return actual_setup == expected_setup


def metadata_path(csv_path: str | Path) -> Path:
    path = Path(csv_path)
    return path.with_suffix(path.suffix + ".meta.json")


def build_simulation_metadata() -> SimulationMetadata:
    root = Path(__file__).resolve().parents[3]
    implementation: tuple[str, ...] = (
        "src/config/config_mcad_parameters.py",
        "src/modules/process/motor_cad_processes/mcad_processing.py",
        "src/modules/simulation/simulation.py",
        "src/modules/simulation/generators/machine_generator.py",
        "src/modules/simulation/generators/map_generator.py",
    )
    payload: SimulationPayload = {
        "metadata_version": 1,
        "base_model_sha256": file_sha256(PATH_MCAD_BASEFILE),
        "mcad_executable_sha256": file_sha256(PATH_MCAD_EXE),
        "integration_version": version("ansys-motorcad-core"),
        "external_dependency_revision": MCAD_DEPENDENCY_REVISION,
        "external_dependency_hashes": {
            str(Path(path).resolve()): file_sha256(path)
            for path in MCAD_DEPENDENCY_FILES
        },
        "csv_columns": list(CSV_PARAMETERS),
        "float_decimal_places": 4,
        "implementation_hashes": {
            name: file_sha256(root / name) for name in implementation
        },
    }
    return {"fingerprint": canonical_hash(payload), "payload": payload}


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
    destination = metadata_path(csv_path)
    temporary: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix="metadata_",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)

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
        if temporary is not None:
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
        raise ValueError("The old file does not contain any result rows.")

    if file_sha256(csv_path) != csv_hash:
        raise ValueError("The CSV file was modified during the audit.")

    document: SimulationMetadata = {
        "fingerprint": expected["fingerprint"],
        "payload": expected["payload"].copy(),
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
