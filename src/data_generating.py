
import pandas as pd
from itertools import product
import config_paths
import process_managing
import ansys.motorcad.core as pymotorcad

class MCADDataGenerator:

    PATH_MOTORCAD_DATA = config_paths.PATH_MCAD_DATA
    PATH_RESULTS = config_paths.PATH_MCAD_DATA_RESULTS
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
            list(
                range(
                    int(row["min_value"]),
                    int(row["max_value"]),
                    int(row["step_size"])
                )
            )
            for _, row in input_parameter.iterrows()
        ]

        input_parameter.drop(columns=["min_value", "max_value", "step_size"], inplace=True)

        return input_parameter

    @staticmethod
    def create_input_df(input_parameter: pd.DataFrame) -> pd.DataFrame:

        input_values_df = pd.DataFrame(
            product(*input_parameter["value_array"].tolist()),
            columns=input_parameter["name"].tolist(),
            dtype=float
        )

        print(f"{len(input_values_df)} value combinations have been created.")
        if len(input_values_df) > 1000000:
            raise OverflowError("Warning: The number of value combinations is too large."
                                " Reducing the step sizes, intervall size or number of parameters.")

        return input_values_df

    def execute_mcad_simulation(self, input_values_df: pd.DataFrame, output_parameters: list[str]) -> pd.DataFrame:
        print("---------Starting MCAD simulation------------")

        output_rows = []
        for _, row in input_values_df.iterrows():
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