from __future__ import annotations

import logging
import os
from multiprocessing.queues import Queue as ProcessQueue
from multiprocessing.util import Finalize
from pathlib import Path

from src.config.config_logging import setup_worker_logging
from src.config.config_paths import PATH_MCAD_BASEFILE
from src.modules.data_managment.temp_files import create_temp_file_copy
from src.modules.process.motor_cad_processes.mcad_processing import MCADProcessor

logger = logging.getLogger(__name__)

_mcad_instance: MCADProcessor | None = None


def init_mcad_instance(
    log_queue: ProcessQueue[logging.LogRecord],
    temp_root: Path | None = None,
) -> None:
    global _mcad_instance

    setup_worker_logging(log_queue)
    logger.info("Initializing MCAD instance in worker PID %s", os.getpid())

    Finalize(
        None,
        shutdown_mcad_instance,
        exitpriority=20,
    )

    if _mcad_instance is None:
        _mcad_instance = MCADProcessor()

    try:
        _mcad_instance.start()
        open_mcad_file(
            _mcad_instance,
            temp_root=temp_root,
        )
    except BaseException:
        try:
            shutdown_mcad_instance()
        except Exception:
            logger.exception("MCAD cleanup failed after worker initialization error")
        raise


def get_mcad_instance() -> MCADProcessor:
    if _mcad_instance is None:
        logger.error("Worker MCAD instance not initialized.")
        raise RuntimeError("Worker MCAD instance not initialized.")

    return _mcad_instance


def shutdown_mcad_instance() -> None:
    global _mcad_instance

    logger.info(
        "Shutting down MCAD in worker PID %d",
        os.getpid(),
    )

    if _mcad_instance is not None:
        _mcad_instance.close()
        _mcad_instance = None


def open_mcad_file(
    mcad_instance: MCADProcessor,
    *,
    temp_root: Path | None = None,
) -> None:
    file_path: Path = create_temp_file_copy(
        Path(PATH_MCAD_BASEFILE),
        temp_root=temp_root,
    )
    mcad_instance.open_motorcad_file(file_path)
