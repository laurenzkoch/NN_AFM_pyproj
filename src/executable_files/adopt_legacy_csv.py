from __future__ import annotations

import argparse
from pathlib import Path

from src.modules.data_managment.simulation_metadata import (
    SimulationMetadata,
    adopt_legacy_csv,
    build_simulation_metadata,
    metadata_path,
)


class Arguments(argparse.Namespace):
    csv_path: Path
    mot_sha256: str
    reason: str
    accept_setup_assumption: bool
    write: bool


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Alte Simulations-CSV pruefen und Metadaten ergaenzen."
    )

    parser.add_argument(
        "csv_path",
        type=Path,
        help="Pfad zur vorhandenen CSV-Datei.",
    )
    parser.add_argument(
        "--mot-sha256",
        required=True,
        help="SHA256 der MOT-Datei, auf der die alten Daten basieren.",
    )
    parser.add_argument(
        "--reason",
        required=True,
        help="Begruendung fuer die angenommene Kompatibilitaet.",
    )
    parser.add_argument(
        "--accept-setup-assumption",
        action="store_true",
        help="Kompatibilitaet des historischen Setups bewusst annehmen.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Metadaten schreiben; andernfalls nur pruefen.",
    )

    arguments: Arguments = Arguments()
    parser.parse_args(namespace=arguments)

    if not arguments.accept_setup_assumption:
        parser.error("--accept-setup-assumption muss angegeben werden.")

    csv_path: Path = arguments.csv_path.resolve()
    expected: SimulationMetadata = build_simulation_metadata()

    adopt_legacy_csv(
        csv_path,
        expected,
        confirmed_mot_hash=arguments.mot_sha256,
        reason=arguments.reason,
        accept_setup_assumption=arguments.accept_setup_assumption,
        write=arguments.write,
    )

    if arguments.write:
        print(f"Metadaten geschrieben: {metadata_path(csv_path)}")
    else:
        print(f"Pruefung erfolgreich: {csv_path}")
        print("Keine Dateien veraendert.")


if __name__ == "__main__":
    main()
