import shutil
from pathlib import Path


def delete_folder(folder: Path) -> None:
    if not folder.exists():
        print(f"folder doesn't exsitst: {folder}")
        return
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a folder: {folder}")

    try:
        shutil.rmtree(folder)
        print(f"Folder deleted: {folder}")
    except Exception as exc:
        raise RuntimeError(f"Failed to delete folder: {folder}") from exc


def delete_file(file: Path) -> None:
    if not file.exists():
        print(f"File doesn't exsitst: {file}")
        return
    if not file.is_file():
        raise NotADirectoryError(f"Path is not a file: {file}")
    file.unlink()
    print(f"File deleted: {file}")
