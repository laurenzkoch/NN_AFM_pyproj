
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
    def write_row_in_csv(path_results: Path, value_row: list[float]) -> None:
        with open(path_results, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";", quotechar='"')
            formatted_value_row = [
                f"{x:.4f}".replace(".", ",") if isinstance(x, float) else x
                for x in value_row
            ]
            writer.writerow(formatted_value_row)