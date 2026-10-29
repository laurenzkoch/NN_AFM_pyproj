from __future__ import annotations

import logging
import os
from multiprocessing.queues import Queue as ProcessQueue
from multiprocessing.util import Finalize
from pathlib import Path
from time import perf_counter

from src.config.config_logging import setup_worker_logging
from src.config.config_paths import PATH_MCAD_BASEFILE
from src.modules.data_managment.temp_files import create_temp_file_copy
from src.modules.process.motor_cad_processes.mcad_errors import (
    MCADCleanupError,
)
from src.modules.process.motor_cad_processes.mcad_processing import MCADProcessor
from src.modules.process.worker_job import protect_worker_process_tree

logger = logging.getLogger(__name__)

_mcad_instance: MCADProcessor | None = None


def init_mcad_instance(
    log_queue: ProcessQueue[logging.LogRecord],
    temp_root: Path | None = None,
) -> None:
    global _mcad_instance

    setup_worker_logging(log_queue)
    started = perf_counter()
    phase = "process_protection"

    logger.info(
        "Worker initialization started: pid=%d temp_root=%s",
        os.getpid(),
        temp_root,
    )

    try:
        protect_worker_process_tree()

        Finalize(
            None,
            shutdown_mcad_instance,
            exitpriority=20,
        )

        phase = "instance_creation"

        if _mcad_instance is None:
            _mcad_instance = MCADProcessor()

        instance = get_mcad_instance()

        phase = "instance_configuration"
        instance.start()

        phase = "model_loading"
        open_mcad_file(
            instance,
            temp_root=temp_root,
        )

    except BaseException:
        logger.exception(
            "Worker initialization failed: phase=%s elapsed_s=%.3f",
            phase,
            perf_counter() - started,
        )

        shutdown_mcad_instance()
        raise

    logger.info(
        "Worker initialization completed: elapsed_s=%.3f",
        perf_counter() - started,
    )


def get_mcad_instance() -> MCADProcessor:
    if _mcad_instance is None:
        logger.error("Worker MCAD instance not initialized.")
        raise RuntimeError("Worker MCAD instance not initialized.")

    return _mcad_instance


def shutdown_mcad_instance(
    *,
    raise_on_error: bool = False,
) -> None:
    global _mcad_instance

    instance = _mcad_instance
    _mcad_instance = None

    if instance is None:
        logger.debug("No registered Motor-CAD instance to close.")
        return

    started = perf_counter()

    logger.info("Worker Motor-CAD cleanup started.")

    try:
        instance.close()

    except Exception as exc:
        logger.exception(
            "Worker Motor-CAD cleanup failed: elapsed_s=%.3f",
            perf_counter() - started,
        )

        if raise_on_error:
            raise MCADCleanupError(
                "Worker could not close its Motor-CAD instance."
            ) from exc

    else:
        logger.info(
            "Worker Motor-CAD cleanup completed: elapsed_s=%.3f",
            perf_counter() - started,
        )


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
