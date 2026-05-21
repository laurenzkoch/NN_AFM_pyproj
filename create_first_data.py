from pathlib import Path

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

    def do_calculation(self) -> None:
        self.mcad.do_magnetic_calculation()

    def export_results(self, suffix: str) -> None:
        filename = f"EMagnetic_{suffix}.csv"
        self.mcad.export_results("EMagnetic", str(self.PATH_RESULTS / filename))

    def get_torque(self) -> float | int | str | bool:
        return self.mcad.get_variable("ShaftTorque")