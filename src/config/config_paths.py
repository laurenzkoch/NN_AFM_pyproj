from pathlib import Path



PATH_MCAD_DATA: Path = Path("D:\\MotorCAD")
PATH_DATA_MCAD_RESULTS: Path = Path(__file__).resolve().parent.parent.parent / "data" / "MCAD_Results"
PATH_TEMP_DATA: Path = Path(__file__).resolve().parent.parent.parent / "data" / "temp_csv"
PATH_TEMP_FILES: Path =  Path("temp_files")
PATH_DATA_PLOTS: Path = Path(__file__).resolve().parent.parent.parent / "data" / "plots"

PATH_eduVPN_EXE: Path = Path("C:\\Program Files\\eduVPN\\Core\\eduVPN.Client.exe")