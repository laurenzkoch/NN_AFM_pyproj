from pathlib import Path
import pandas as pd
import os

import ansys.motorcad.core as pymotorcad

class MCAD:

    PATH_MOTORCAD_DATA = Path("D:\\MotorCAD")
    PATH_RESULTS = Path("D:\\MotorCAD\\Results")
    PATH_RESULTS.mkdir(exist_ok=True)

    def __init__(self):
        self.mcad = pymotorcad.MotorCAD()
        self.mcad.set_visible(True)

    def open_motorcad_file(self, filename: str) -> None:
        file_path = self.PATH_MOTORCAD_DATA / filename
        if not file_path.exists():
            raise FileNotFoundError(f"file not found: {file_path}")
        self.mcad.load_from_file(str(file_path))

    def set_peak_current(self, peak_current_ampere: float) -> None:
        self.mcad.set_variable("PeakCurrent", peak_current_ampere)

    def set_phase_advance(self, phase_advance_degree: float) -> None:
        self.mcad.set_variable("PhaseAdvance", phase_advance_degree)

    def do_calculation(self) -> None:
        self.mcad.do_magnetic_calculation()

    def export_results(self, suffix: str) -> None:
        filename = f"EMagnetic_{suffix}.csv"
        path_results = str(self.PATH_RESULTS / filename)
        self.mcad.export_results("EMagnetic", path_results)

    def get_torque(self) -> float:
        try:
            shaft_torque = float(self.mcad.get_variable("ShaftTorque"))
            return shaft_torque
        except ValueError:
            raise ValueError("Failed to convert ShaftTorque to float")

    """
    def get_flux(self) -> float:
        try:
            flux = float(self.mcad.get_variable("Flux"))
            return flux
        except ValueError:
            raise ValueError("Failed to convert Flux to float")
    """

    def generate_data_set_shaft_torque(self, parameter: str, parameter_vektor: list[int]) -> pd.DataFrame:
        self.open_motorcad_file(filename="base_first_data.mot")

        parameter_values: list[float] = []
        shaft_torque: list[float] = []

        for parameter_value in parameter_vektor:
            self.mcad.set_variable(parameter, parameter_value)
            self.do_calculation()
            parameter_values.append(parameter_value)
            shaft_torque.append(self.get_torque())

        torque_results_df = pd.DataFrame({
            parameter: parameter_values,
            "Torque": shaft_torque
        })

        return torque_results_df

    def export_data_set_shaft_torque(self, torque_results_df: pd.DataFrame, parameter: str) -> None:
        filename = f"torque_results_{parameter}_variation.csv"
        path_results = str(self.PATH_RESULTS / filename)
        torque_results_df.to_csv(path_results, index=False, sep=";", decimal=",", float_format="%.4f", encoding="utf-8")
        os.startfile(path_results)