from datetime import UTC, datetime
from pathlib import Path

from src.config.config_paths import PATH_DATA_MCAD_RESULTS
from src.modules.data_managment.experiment_utils import file_sha256
from src.modules.data_managment.simulation_metadata import (
    SimulationMetadata,
    build_simulation_metadata,
    metadata_path,
    read_metadata,
    validate_result_csv,
    write_metadata,
)

CSV_PATH: Path = (
    PATH_DATA_MCAD_RESULTS
    / "Results_EMagnetic_simulation_10_10_5_GenType-v2_12.10.2026_21-45-45_CEST.csv"
)

MOT_SHA256: str = "553b143c3e4c0861877c8e7284e6f9908e22c9d0040c6a100fec9c3ff029f852"

REASON: str = (
    "Aktuellen CSV-Stand bewusst neu übernommen. "
    "Daten stammen aus derselben MOT-Datei und demselben Simulationssetup."
)


def main() -> None:
    expected: SimulationMetadata = build_simulation_metadata()

    if MOT_SHA256.strip().lower() != expected["payload"]["base_model_sha256"]:
        raise ValueError("Die MOT-Datei passt nicht zur aktuellen Konfiguration.")

    csv_hash: str = file_sha256(CSV_PATH)

    frame = validate_result_csv(CSV_PATH)
    if frame.empty:
        raise ValueError("Die CSV enthält keine Ergebniszeilen.")

    document: SimulationMetadata = {
        "fingerprint": expected["fingerprint"],
        "payload": expected["payload"],
        "adoption": {
            "kind": "legacy_assumption",
            "reason": REASON,
            "accepted_at_utc": datetime.now(UTC).isoformat(),
            "csv_sha256": csv_hash,
            "original_setup_verified": False,
        },
    }

    if file_sha256(CSV_PATH) != csv_hash:
        raise RuntimeError("Die CSV wurde während der Prüfung verändert.")

    write_metadata(CSV_PATH, document)

    saved: SimulationMetadata = read_metadata(CSV_PATH)
    if saved != document:
        raise RuntimeError("Gespeicherte Metadaten stimmen nicht überein.")

    print(f"Metadaten neu geschrieben: {metadata_path(CSV_PATH)}")
    print("Aktueller CSV-Hash gespeichert. CSV unverändert.")


if __name__ == "__main__":
    main()
