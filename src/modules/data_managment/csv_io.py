import logging
from pathlib import Path
import csv
import pandas as pd

logger = logging.getLogger(__name__)

def create_csv_file(path: Path, columns_list: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        logger.warning("CSV file already existing. File will not be overwritten.")
        return

    with open(path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";", quotechar='"')
        writer.writerow(columns_list)


def import_data_from_csv(path_file: Path) -> pd.DataFrame:
    data_df = pd.read_csv(path_file, encoding='utf-8', sep=";", decimal=",")
    return data_df


def write_row_in_csv(
        path_results: Path,
        value_row: list[float]
) -> None:

    with open(path_results, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";", quotechar='"')
        formatted_value_row = [
            f"{x:.4f}".replace(".", ",") if isinstance(x, float) else x
            for x in value_row
        ]
        writer.writerow(formatted_value_row)


def write_from_csv_in_csv(
        input_folder: Path,
        output_file: Path,
) -> None:

    csv_files = sorted(input_folder.glob(pattern="*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_folder}")

    with open(output_file, mode="a", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile, delimiter=";", quotechar='"')

        for csv_file in csv_files:

            with open(csv_file, mode="r", newline="", encoding="utf-8") as infile:
                reader = csv.reader(infile, delimiter=";", quotechar='"')

                for row in reader:
                    writer.writerow(row)