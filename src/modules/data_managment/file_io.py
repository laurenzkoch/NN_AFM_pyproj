import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def delete_folder(folder: Path) -> None:
    if not folder.exists():
        logger.debug("Folder does not exist: %s", folder)
        return

    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a folder: {folder}")

    try:
        shutil.rmtree(folder)
    except OSError as exc:
        raise RuntimeError(f"Failed to delete folder: {folder}") from exc

    logger.info("Folder deleted: %s", folder)


def delete_file(file: Path) -> None:
    if not file.exists():
        logger.debug("File does not exist: %s", file)
        return

    if file.is_dir():
        raise IsADirectoryError(f"Path is a directory: {file}")

    if not file.is_file():
        raise ValueError(f"Path is not a regular file: {file}")

    file.unlink()
    logger.info("File deleted: %s", file)
