import time

from src.modules import process_managing
import ansys.motorcad.core as pymotorcad
from src.config import config_paths

class MCADProcessor(process_managing.ProcessManager):

    def __init__(self):
        super().__init__()
        print(f"-----------Starting MCAD: Connection to licence server should have been initialized----------")
        self.mcad_process = pymotorcad.MotorCAD()
        self.mcad_process.set_visible(True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.mcad_process is not None:
                try:
                    self.mcad_process.quit()
                    time.sleep(2)
                finally:
                    self.mcad_process = None
        finally:
            pass


    def open_motorcad_file(self, filename: str = "base_first_data.mot") -> None:
       file_path = config_paths.PATH_MCAD_DATA / filename
       if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

       self.mcad_process.load_from_file(str(file_path))


    def set_value_in_mcad(self, variable_name: str, value: float) -> None:
        try:
            self.mcad_process.set_variable(variable_name, value)
        except Exception as exc:
            raise ValueError(f"Failed to set variable {variable_name} to value {value}.") from exc


    def get_value_from_mcad(self, variable_name: str) -> float:
        try:
            return float(self.mcad_process.get_variable(variable_name))
        except Exception as exc:
            raise ValueError(f"Failed to get variable {variable_name} from MotorCAD.") from exc


    def do_calculation(self) -> bool:
        try:
            self.mcad_process.do_magnetic_calculation()
            return True
        except Exception as exc:
            print("Warning: Motor-CAD rejected this design point.")
            print(f"Motor-CAD error: {exc}")
            return False