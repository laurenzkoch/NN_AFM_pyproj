from pathlib import Path

import pandas as pd

from src.config import config_paths


def find_nan_values(csv_path: Path) -> None:
    data = pd.read_csv(
        csv_path,
        encoding="utf-8",
        sep=";",
        decimal=","
    )

    nan_positions = data.isna()

    if not nan_positions.any().any():
        print("Keine NaN-Werte gefunden.")
        return

    print("NaN-Werte gefunden:")

    for row_index, row in nan_positions.iterrows():
        columns_with_nan = row[row].index.tolist()

        if columns_with_nan:
            csv_line_number = row_index + 2
            print(f"CSV-Zeile {csv_line_number}: {columns_with_nan}")


def main() -> None:
    csv_path = config_paths.PATH_DATA_MCAD_RESULTS / "Results_EMagnetic_simulation_radial-variation_28.07.2026_21-00-39_CEST.csv"


    find_nan_values(csv_path)


if __name__ == "__main__":
    main()