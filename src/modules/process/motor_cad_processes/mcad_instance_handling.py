from modules.process.motor_cad_processes import mcad_processing
from config.config_paths import PATH_MCAD_BASEFILE
from modules.data_managment.temp_files import create_temp_file_copy

import atexit
from pathlib import Path


_mcad_instance: mcad_processing.MCADProcessor | None = None


def init_mcad_instance() -> None:
    global _mcad_instance

    atexit.register(shutdown_mcad_instance)

    if _mcad_instance is None:
        _mcad_instance = mcad_processing.MCADProcessor()
    if _mcad_instance is not None:
        open_mcad_file(_mcad_instance)


def get_mcad_instance() -> mcad_processing.MCADProcessor:
    global _mcad_instance
    if _mcad_instance is None:
        raise RuntimeError("Worker MCAD instance not initialized")
    return _mcad_instance


def shutdown_mcad_instance() -> None:
    global _mcad_instance
    if _mcad_instance is not None:
        _mcad_instance.__exit__(None, None, None)
        _mcad_instance = None


def open_mcad_file(mcad_instance: mcad_processing.MCADProcessor) -> None:

    file_path = create_temp_file_copy(
        Path(PATH_MCAD_BASEFILE)
    )
    mcad_instance.open_motorcad_file(file_path)