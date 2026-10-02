
import os
import logging
import atexit
from pathlib import Path
from multiprocessing import Queue

from src.modules.process.motor_cad_processes import mcad_processing
from src.config.config_paths import PATH_MCAD_BASEFILE
from src.modules.data_managment.temp_files import create_temp_file_copy
from src.config.config_logging import setup_worker_logging

logger = logging.getLogger(__name__)

_mcad_instance: mcad_processing.MCADProcessor | None = None


def init_mcad_instance(log_queue: Queue) -> None:
    global _mcad_instance

    setup_worker_logging(log_queue)
    logger.info(f"Initialization MCAD instance: Connection to licence server should have been initialized")

    atexit.register(shutdown_mcad_instance)

    if _mcad_instance is None:
        _mcad_instance = mcad_processing.MCADProcessor()
    if _mcad_instance is not None:
        _mcad_instance.start()
        open_mcad_file(_mcad_instance)


def get_mcad_instance() -> mcad_processing.MCADProcessor:
    global _mcad_instance
    if _mcad_instance is None:
        logger.exception("Worker MCAD instance not initialized")
        raise RuntimeError("Worker MCAD instance not initialized")
    return _mcad_instance


def shutdown_mcad_instance() -> None:
    global _mcad_instance

    logger.info(f"Shutting down MCAD in worker PID {os.getpid()}")

    if _mcad_instance is not None:
        _mcad_instance.close()
        _mcad_instance = None


def open_mcad_file(mcad_instance: mcad_processing.MCADProcessor) -> None:

    file_path = create_temp_file_copy(
        Path(PATH_MCAD_BASEFILE)
    )
    mcad_instance.open_motorcad_file(file_path)