from pathlib import Path



PATH_MCAD_DATA: Path = Path("D:\\MotorCAD")
PATH_DATA: Path = Path(__file__).resolve().parent.parent.parent / "data"
PATH_DATA_MCAD_RESULTS: Path = PATH_DATA / "MCAD_Results"
PATH_TEMP_DATA: Path = PATH_DATA / "temp_csv"
PATH_TEMP_FILES: Path =  Path("temp_files")
PATH_DATA_PLOTS: Path = PATH_DATA / "plots"

PATH_eduVPN_EXE: Path = Path("C:\\Program Files\\eduVPN\\Core\\eduVPN.Client.exe")