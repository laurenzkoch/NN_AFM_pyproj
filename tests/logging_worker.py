import logging
import os

logger = logging.getLogger("src.tests.logging_worker")


def write_markers() -> int:
    logger.debug("WORKER_DEBUG_TEST")
    logger.info("WORKER_INFO_TEST")
    logger.error("WORKER_ERROR_TEST")

    return os.getpid()
