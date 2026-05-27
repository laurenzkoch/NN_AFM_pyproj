from pathlib import Path
import pandas as pd
import os
from itertools import product
import matplotlib.pyplot as plt

import ansys.motorcad.core as pymotorcad

import utils

class MCAD:

    PATH_MOTORCAD_DATA = Path("D:\\MotorCAD")
    PATH_RESULTS = Path("D:\\MotorCAD\\Results")
    PATH_RESULTS.mkdir(exist_ok=True)

    def __init__(self):
        self.mcad = None
        self.eduVPN = None

        self.eduVPN = utils.start_process("C:\\Program Files\\eduVPN\\Core\\eduVPN.Client.exe")

        self.mcad = pymotorcad.MotorCAD()
        self.mcad.set_visible(True)

    def close(self) -> None:
        try:
            if self.mcad is not None:
                try:
                    self.mcad.quit()
                finally:
                    self.mcad = None
        finally:
            if self.eduVPN is not None:
                utils.stop_process(self.eduVPN)
                self.eduVPN = None

            utils.stop_process_by_name("eduVPN.Client.exe", timeout_seconds=15.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


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

    def generate_data_set_shaft_torque(self, parameter_dict: dict[str, list[int]]) -> pd.DataFrame:

        torque_results_df: pd.DataFrame = pd.DataFrame(
            columns=list(parameter_dict.keys()) + ["ShaftTorque"]
        )

        rows = []
        for combination in product(*parameter_dict.values()):
            for parameter_name, value in zip(parameter_dict.keys(), combination):
                self.mcad.set_variable(parameter_name, value)
            self.do_calculation()
            row = {
                **dict(zip(parameter_dict.keys(), combination)),
                "ShaftTorque": self.get_torque(),
            }
            rows.append(row)
            torque_results_df = pd.DataFrame(rows)

        return torque_results_df

    def export_data_set_shaft_torque(self, torque_results_df: pd.DataFrame) -> None:
        filename = f"torque_results_variation.csv"
        path_results = str(self.PATH_RESULTS / filename)
        torque_results_df.to_csv(path_results, index=False, sep=";", decimal=",", float_format="%.4f", encoding="utf-8")
        os.startfile(path_results)

    def create_plot(self, torque_results_df: pd.DataFrame) -> None:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        plot = ax.scatter(
            torque_results_df["PeakCurrent"],
            torque_results_df["PhaseAdvance"],
            torque_results_df["ShaftTorque"],
            c=torque_results_df["ShaftTorque"],
            cmap="viridis",
        )

        ax.set_xlabel("PeakCurrent")
        ax.set_ylabel("PhaseAdvance")
        ax.set_zlabel("ShaftTorque")

        fig.colorbar(plot, ax=ax, label="ShaftTorque")
        fig.savefig(str(self.PATH_RESULTS / "torque_results_variation.png"), dpi=300)

        plt.show()
