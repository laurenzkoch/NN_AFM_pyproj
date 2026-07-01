import shutil
import tempfile

import pandas as pd
from pathlib import Path
import csv

class DataManager:

    def __init__(self) -> None:
        pass

    @staticmethod
    def import_data_from_csv(
            path_data: Path,
            filename: str
    ) -> pd.DataFrame:
        data_df = pd.read_csv(path_data / filename, encoding='utf-8', sep=";", decimal=",")
        return data_df


    @staticmethod
    def split_data(
            data_df: pd.DataFrame,
            number_of_outputs: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        x_df = data_df.iloc[:, :-number_of_outputs]
        y_df = data_df.iloc[:, -number_of_outputs:]
        return x_df, y_df

    @staticmethod
    def split_chunks(values, number_of_chunks):
        chunks = [[] for _ in range(number_of_chunks)]

        for index, value in enumerate(values):
            chunks[index % number_of_chunks].append(value)

        return chunks

    @staticmethod
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

    @staticmethod
    def create_csv_file(path: Path, columns_list: list[str]):
        print(f"--------- Creating CSV file ------------")

        with open(path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";", quotechar='"')
            writer.writerow(columns_list)


    @staticmethod
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

    @staticmethod
    def delete_folder(folder: Path) -> None:
        if not folder.exists():
            print(f"folder doesn't exsitst: {folder}")
            return

        if not folder.is_dir():
            raise NotADirectoryError(f"Path is not a folder: {folder}")

        shutil.rmtree(folder)
        print(f"folder deleted: {folder}")

    @staticmethod
    def create_temp_file(source_path: Path) -> Path:

        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {source_path}")

        with tempfile.NamedTemporaryFile(
                dir=source_path.parent,
                prefix=f"{source_path.stem}_",
                suffix=source_path.suffix,
                delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)

        shutil.copy2(source_path, temp_path)

        return temp_path

    @staticmethod
    def delete_temp_file_and_related_folder(temp_file_path: Path) -> None:
        related_folder = temp_file_path.with_suffix("")

        if temp_file_path.exists():
            if not temp_file_path.is_file():
                raise IsADirectoryError(f"Path is not a file: {temp_file_path}")
            temp_file_path.unlink()
            print(f"temp file deleted: {temp_file_path}")

        DataManager.delete_folder(related_folder)
