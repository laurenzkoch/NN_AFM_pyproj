import shutil
import tempfile
from pathlib import Path

from src.config.config_paths import PATH_TEMP_FILES


def create_temp_file_copy(
    source_path: Path,
    *,
    temp_root: Path | None = None,
) -> Path:
    source_path = source_path.resolve()

    if not source_path.is_file():
        raise FileNotFoundError(f"File not found: {source_path}")

    parent_directory: Path = (
        temp_root if temp_root is not None else source_path.parent / PATH_TEMP_FILES
    )
    parent_directory.mkdir(parents=True, exist_ok=True)

    instance_directory: Path = Path(
        tempfile.mkdtemp(
            dir=parent_directory,
            prefix=f"{source_path.stem}_",
        )
    )
    destination: Path = instance_directory / source_path.name

    try:
        shutil.copy2(source_path, destination)
    except BaseException:
        shutil.rmtree(instance_directory, ignore_errors=True)
        raise

    return destination


def create_temp_csv_file(source_path: Path) -> Path:
    path_temp_csv_folder = Path(source_path / "temp_csv")
    path_temp_csv_folder.mkdir(exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=path_temp_csv_folder, prefix="temp_", suffix=".csv", delete=False
    ) as temp_file:
        temp_csv_path = Path(temp_file.name)
        return temp_csv_path
