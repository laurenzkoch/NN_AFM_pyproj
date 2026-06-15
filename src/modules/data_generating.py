import csv
from typing import Callable
import pandas as pd
from itertools import product
import time
import numpy as np
from pathlib import Path

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
        self.input_value_array_df = pd.DataFrame()

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


    def take_input_parameter(self, input_df: pd.DataFrame):
        print("---------Preparing input parameter------------")
        self.input_value_array_df, self.number_of_combinations = self.generate_input_value_array_df(input_df)


    @staticmethod
    def generate_input_value_array_df(input_parameter_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:

        input_parameter_df["value_array"] = [
            np.arange(
                float(row["min_value"]),
                float(row["max_value"]),
                float(row["step_size"])
            ).tolist()
            for _, row in input_parameter_df.iterrows()
        ]

        input_parameter_df.drop(columns=["min_value", "max_value", "step_size"], inplace=True)

        number_of_combinations = 1
        for i in range(len(input_parameter_df["value_array"]) - 1):
            number_of_combinations = input_parameter_df["value_array"].iloc[i].__len__() * number_of_combinations

        print(f"{number_of_combinations} combinations will be created.")

        return input_parameter_df, number_of_combinations


    def get_combination_value(self, variable_name: str, combination_tuple: tuple) -> float:
        variable_index = self.input_value_array_df["name"].tolist().index(variable_name)
        return combination_tuple[variable_index]


    def test_boundary_conditions(self, combination_values: tuple) -> bool:
        test_passed = True

        if ("Slot_Depth" in self.input_value_array_df["name"]
                and "Stator_Lam_Length" in self.input_value_array_df["name"]):
            if self.get_combination_value(variable_name="Slot_Depth", combination_tuple=combination_values) >= self.get_combination_value(variable_name="Stator_Lam_Length", combination_tuple=combination_values):
                test_passed = False
        if ("Motor_Length" in self.input_value_array_df["name"]
                and "Stator_Lam_Length" in self.input_value_array_df["name"]):
            if self.get_combination_value(variable_name="Motor_Length", combination_tuple=combination_values) <= self.get_combination_value(variable_name="Stator_Lam_Length", combination_tuple=combination_values):
                test_passed = False
        if ("Magnet_Length" in self.input_value_array_df["name"]
                and "Airgap" in self.input_value_array_df["name"]):
            if self.get_combination_value(variable_name="Magnet_Length", combination_tuple=combination_values) <= self.get_combination_value(variable_name="Airgap", combination_tuple=combination_values):
                test_passed = False
        if ("Magnet_Length" in self.input_value_array_df["name"]
                and "Stator_Lam_Length" in self.input_value_array_df["name"]):
            if self.get_combination_value(variable_name="Magnet_Length", combination_tuple=combination_values) >= self.get_combination_value(variable_name="Stator_Lam_Length", combination_tuple=combination_values):
                test_passed = False
        if ("Stator_Bore" in self.input_value_array_df["name"]
                and "Stator_Lam_Dia" in self.input_value_array_df["name"]):
            if self.get_combination_value(variable_name="Stator_Bore", combination_tuple=combination_values) >= self.get_combination_value(variable_name="Stator_Lam_Dia", combination_tuple=combination_values):
                test_passed = False

        return test_passed


    @staticmethod
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


    def placeholdername(self, output_parameters: list, running_number: int = 1) -> Path:

        filename = f"Results_EMagnetic_simulation_{running_number}.csv"
        path_results = self.PATH_RESULTS / filename

        if path_results.exists():
            print(f"File {filename} already exists. Incrementing running number and trying again.")
            running_number += 1
            self.placeholdername(
                output_parameters=output_parameters,
                running_number=running_number
            )
            return path_results

        with open(path_results, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";", quotechar='"')
            writer.writerow(self.input_value_array_df["name"].tolist() + output_parameters)


        drop_counter = 0
        comb_counter = 0
        progress_in_percent = 0.0
        tracking_progress = self.track_progress(self.number_of_combinations)

        for combination in product(*self.input_value_array_df["value_array"].tolist()):


            if comb_counter > 1000000:
                raise OverflowError("Warning: The number of value combinations is too large."
                                    " Reducing the step sizes, intervall size or number of parameters.")

            if self.test_boundary_conditions(combination):
                value_row = []

                for variable in self.input_value_array_df["name"]:
                    value = self.get_combination_value(variable_name=variable, combination_tuple=combination)
                    value_row.append(value)
                    try:
                        self.mCAD_process.set_variable(variable, value)
                    except Exception as exc:
                        raise ValueError(f"Failed to set variable {variable} to value {value}.") from exc
                try:
                    self.mCAD_process.do_magnetic_calculation()
                except Exception as exc:
                    print(value_row)
                    raise ValueError(f"Failed to execute MCAD calculation.") from exc

                for variable in output_parameters:
                    try:
                        value_row.append(float(self.mCAD_process.get_variable(variable)))
                    except Exception as exc:
                        raise ValueError(f"Failed to get variable {variable} from MCAD simulation.") from exc

                with open(path_results, mode="a", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file, delimiter=";", quotechar='"')
                    writer.writerow(value_row)

                comb_counter += 1
            else:
                drop_counter += 1

            progress_in_percent = tracking_progress(
                drop_counter + comb_counter,
                progress_percent=progress_in_percent
            )


        print(f"{drop_counter} points dropped. {comb_counter} points saved. Input points saved in {filename}")

        return path_results



"""    
    def create_input_csv(self, input_parameters_df: pd.DataFrame, running_number: int = 1) -> Path:

        filename = f"Results_EMagnetic_simulation_{running_number}.csv"
        path_results = self.PATH_RESULTS / filename

        if path_results.exists():
            print(f"File {filename} already exists. Incrementing running number and trying again.")
            running_number += 1
            self.create_input_csv(
                input_parameters_df=input_parameters_df,
                running_number=running_number
            )
            return path_results

        def test_boundary_conditions(combination_values: tuple) -> bool:
            test_passed = True

            def get_combination_value(column_name: str) -> float:
                column_index = input_parameters_df["name"].tolist().index(column_name)
                return combination_values[column_index]

            if "Slot_Depth" in input_parameters_df["name"] and "Stator_Lam_Length" in input_parameters_df["name"]:
                if get_combination_value("Slot_Depth") > get_combination_value("Stator_Lam_Length"):
                    test_passed = False
            if "Motor_Length" in input_parameters_df["name"] and "Stator_Lam_Length" in input_parameters_df["name"]:
                if get_combination_value("Motor_Length") < get_combination_value("Stator_Lam_Length"):
                    test_passed = False
            if "Magnet_Length" in input_parameters_df["name"] and "Airgap" in input_parameters_df["name"]:
                if get_combination_value("Magnet_Length") < get_combination_value("Airgap"):
                    test_passed = False
            if "Magnet_Length" in input_parameters_df["name"] and "Stator_Lam_Length" in input_parameters_df["name"]:
                if get_combination_value("Magnet_Length") > get_combination_value("Stator_Lam_Length"):
                    test_passed = False
            if "Stator_Bore" in input_parameters_df["name"] and "Stator_Lam_Dia" in input_parameters_df["name"]:
                if get_combination_value("Stator_Bore") > get_combination_value("Stator_Lam_Dia"):
                    test_passed = False

            return test_passed


        with open(path_results, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";", quotechar='"')
            writer.writerow(input_parameters_df["name"].tolist())

        drop_counter = 0
        comb_counter = 0
        for combination in product(*input_parameters_df["value_array"].tolist()):
            if comb_counter > 1000000:
                raise OverflowError("Warning: The number of value combinations is too large."
                                    " Reducing the step sizes, intervall size or number of parameters.")

            if test_boundary_conditions(combination):
                with open(path_results, mode="a", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file, delimiter=";", quotechar='"')
                    writer.writerow(combination)
                    comb_counter += 1
            else:
                drop_counter += 1

        print(f"{drop_counter} points dropped. {comb_counter} points saved. Input points saved in {filename}")

        return path_results

    def execute_mcad_simulation(self, path_results: Path, output_parameters: list[str]) -> pd.DataFrame:
        print("---------Starting MCAD simulation------------")

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

        input_values_df = pd.read_csv(
            path_results,
            usecols=[0, row],
            sep=";",
            decimal=","
        )
        output_rows = []
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


"""