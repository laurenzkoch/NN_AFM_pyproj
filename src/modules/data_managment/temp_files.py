import shutil
import tempfile

from src.config.config_paths import *


def create_temp_file_copy(source_path: Path) -> Path:
    if not source_path.exists():
        raise FileNotFoundError(f"File not found: {source_path}")

    path_parent = source_path.parent / PATH_TEMP_FILES
    path_parent.mkdir(exist_ok=True)

    with tempfile.NamedTemporaryFile(
            dir=path_parent,
            prefix=f"{source_path.stem}_",
            suffix=source_path.suffix,
            delete=False
    ) as temp_file:
        temp_path = Path(temp_file.name)

    shutil.copy2(source_path, temp_path)

    return temp_path


def create_temp_csv_file(source_path: Path) -> Path:
    path_temp_csv_folder = Path(source_path / "temp_csv")
    path_temp_csv_folder.mkdir(exist_ok=True)

    with tempfile.NamedTemporaryFile(
            dir=path_temp_csv_folder,
            prefix=f"temp_",
            suffix=".csv",
            delete=False
    ) as temp_file:
        temp_csv_path = Path(temp_file.name)
        return temp_csv_path
