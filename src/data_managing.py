
import pandas as pd
from pathlib import Path

class DataManager:

    def __init__(self) -> None:
        pass

    def import_data_from_csv(self, path_data: Path, filename: str) -> pd.DataFrame:
        data_df = pd.read_csv(path_data / filename, encoding='utf-8', sep=";", decimal=",")
        return data_df

    def strip_data(self, data_df: pd.DataFrame, number_of_outputs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        x_df = data_df.iloc[:, :-number_of_outputs]
        y_df = data_df.iloc[:, -number_of_outputs:]
        return x_df, y_df