import logging

logger = logging.getLogger(__name__)


def write_markers() -> None:
    logger.info("WORKER_INFO_TEST")
    logger.error("WORKER_ERROR_TEST")
