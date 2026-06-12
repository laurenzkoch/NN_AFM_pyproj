from typing import Callable
import pandas as pd
from itertools import product
import time
import numpy as np

from src.config import config_paths
from src.modules import process_managing
import ansys.motorcad.core as pymotorcad

class MCADDataGenerator:

    PATH_MOTORCAD_DATA = config_paths.PATH_MCAD_DATA
    PATH_RESULTS = config_paths.PATH_DATA_MCAD_RESULTS
    PATH_RESULTS.mkdir(exist_ok=True)

    def __init__(self, start_eduvpn: bool = True):
        self.processmanager = process_managing.ProcessManager()
        self.eduVPN_process = None

        if start_eduvpn:
            print("-----------Starting eduVPN----------")
            self.eduVPN_process = self.processmanager.start_process(config_paths.PATH_eduVPN_EXE)

        print("-----------Starting MCAD----------")
        self.mCAD_process = pymotorcad.MotorCAD()
        self.mCAD_process.set_visible(False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.mCAD_process is not None:
                try:
                    self.mCAD_process.quit()
                finally:
                    self.mCAD_process = None
        finally:
            if self.eduVPN_process is not None:
                self.processmanager.stop_process(self.eduVPN_process)
                self.eduVPN_process  = None


    def open_motorcad_file(self, filename: str) -> None:

       file_path = self.PATH_MOTORCAD_DATA / filename
       if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

       self.mCAD_process.load_from_file(str(file_path))


    def take_input_parameter(self, input_df: pd.DataFrame) -> pd.DataFrame:
        print("---------Preparing input parameter------------")
        input_value_array_df = self.generate_input_value_array(input_df)
        input_values_df = self.create_input_df(input_value_array_df)

        return input_values_df


    @staticmethod
    def generate_input_value_array(input_parameter: pd.DataFrame) -> pd.DataFrame:

        input_parameter["value_array"] = [
            np.arange(
                float(row["min_value"]),
                float(row["max_value"]),
                float(row["step_size"])
            ).tolist()
            for _, row in input_parameter.iterrows()
        ]

        input_parameter.drop(columns=["min_value", "max_value", "step_size"], inplace=True)

        number_of_combinations = 1
        for i in range(len(input_parameter["value_array"]) - 1):
            number_of_combinations = input_parameter["value_array"].iloc[i].__len__() * number_of_combinations
        print(f"{number_of_combinations} combinations will be created.")

        return input_parameter

    @staticmethod
    def create_input_df(input_parameter: pd.DataFrame) -> pd.DataFrame:

        input_values_df = pd.DataFrame(
            product(*input_parameter["value_array"].tolist()),
            columns=input_parameter["name"].tolist(),
            dtype=float
        )
        print(f"{len(input_values_df)} value combinations have been created.")

        def test_boundary_conditions(input_df: pd.DataFrame) -> pd.DataFrame:
            drop_counter = 0
            for index, row in input_df.iterrows():
                if "Slot_Depth" in input_df.columns and "Stator_Lam_Length" in input_df.columns:
                    if row["Slot_Depth"] > row["Stator_Lam_Length"]:
                        input_df = input_df.drop(index)
                        drop_counter += 1
                if "Motor_Length" in input_df.columns and "Stator_Lam_Length" in input_df.columns:
                    if row["Motor_Length"] < row["Stator_Lam_Length"]:
                        input_df = input_df.drop(index)
                        drop_counter += 1
                if "Magnet_Length" in input_df.columns and "Airgap" in input_df.columns:
                    if row["Magnet_Length"] < row["Airgap"]:
                        input_df = input_df.drop(index)
                        drop_counter += 1
                if "Magnet_Length" in input_df.columns and "Stator_Lam_Length" in input_df.columns:
                    if row["Magnet_Length"] > row["Stator_Lam_Length"]:
                        input_df = input_df.drop(index)
                        drop_counter += 1
                if "Stator_Bore" in input_df.columns and "Stator_Lam_Dia" in input_df.columns:
                    if row["Stator_Bore"] > row["Stator_Lam_Dia"]:
                        input_df = input_df.drop(index)
                        drop_counter += 1
                print(f"{drop_counter} value combinations have been dropped due to boundary conditions.")
            return input_df

        input_values_df = test_boundary_conditions(input_values_df)
        input_values_df = input_values_df.reset_index(drop=True)

        if len(input_values_df) > 1000000:
            raise OverflowError("Warning: The number of value combinations is too large."
                                " Reducing the step sizes, intervall size or number of parameters.")

        return input_values_df

    def execute_mcad_simulation(self, input_values_df: pd.DataFrame, output_parameters: list[str]) -> pd.DataFrame:
        print("---------Starting MCAD simulation------------")

        output_rows = []

        def track_progress(number_of_total_progress_steps: int) -> Callable[..., float]:
            starttime = time.time()

            def inner_track_progress(progress_step: int, progress_percent: float = 0.0) -> float:
                progress_percent_new = progress_step / number_of_total_progress_steps * 100

                elapsedtime = time.time() - starttime
                remainingtime = (elapsedtime / progress_step * (number_of_total_progress_steps - progress_step))
                hours = int(remainingtime // 3600)
                minutes = int((remainingtime % 3600) // 60)
                seconds = remainingtime % 60

                if progress_percent_new >= progress_percent + 10 or progress_percent_new == 0:
                    print(f"{progress_percent_new:.1f}% done. Remaining time: {hours}h {minutes}min {seconds:.2f}ss")
                return progress_percent_new

            return inner_track_progress

        total_simulations = len(input_values_df)
        tracking_progress = track_progress(total_simulations)
        progress_in_percent = 0.0

        for index, (_, row) in enumerate(input_values_df.iterrows(), start=1):

            progress_in_percent = tracking_progress(index, progress_percent=progress_in_percent)

            for variable in input_values_df.columns:
                value = float(row[str(variable)])
                try:
                    self.mCAD_process.set_variable(variable, value)
                except Exception as exc:
                    raise ValueError(f"Failed to set variable {variable} to value {value}.") from exc

            try:
                self.mCAD_process.do_magnetic_calculation()
            except Exception as exc:
                raise ValueError(f"Failed to execute MCAD calculation.") from exc

            output_row = {}
            for variable in output_parameters:
                try:
                    output_row[variable] = float(self.mCAD_process.get_variable(variable))
                except Exception as exc:
                    raise ValueError(f"Failed to get variable {variable} from MCAD simulation.") from exc

            output_rows.append(output_row)

        output_values_df = pd.DataFrame(output_rows, columns=output_parameters)

        results_df = pd.concat(
            [
                input_values_df.reset_index(drop=True),
                output_values_df.reset_index(drop=True)
            ],
            axis=1
        )

        print("---------MCAD simulation finished------------")
        print(f"{len(results_df)} datapoints have been created.")

        return results_df

    def export_results(self, results_df:pd.DataFrame, running_number: int = 1) -> None:
        filename = f"Results_EMagnetic_simulation_{running_number}.csv"
        path_results = self.PATH_RESULTS / filename

        if path_results.exists():
            print(f"File {filename} already exists. Incrementing running number and trying again.")
            running_number += 1
            self.export_results(results_df, running_number)
            return

        results_df.to_csv(path_results, index=False, sep=";", decimal=",", float_format="%.4f", encoding="utf-8")
        print(f"Results saved as {filename}")