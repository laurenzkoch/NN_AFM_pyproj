from __future__ import annotations

import logging
import logging.handlers
from datetime import datetime
from multiprocessing.queues import Queue as ProcessQueue
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from src.config.config_paths import PATH_LOGS

LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d"
    " | %(levelname)s"
    " | pid=%(process)d"
    " | process=%(processName)s"
    " | thread=%(threadName)s"
    " | %(name)s"
    " | %(message)s"
)

LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_application_logging() -> None:
    PATH_LOGS.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y%m%d_%H%M%S_%f")

    run_id = uuid4().hex[:8]
    log_path = PATH_LOGS / f"{timestamp}_{run_id}_application.log"

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(
        log_path,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            console_handler,
            file_handler,
        ],
        force=True,
    )

    logging.getLogger("src").setLevel(logging.DEBUG)


def setup_main_logging(
    log_queue: ProcessQueue[logging.LogRecord],
) -> logging.handlers.QueueListener:
    PATH_LOGS.mkdir(parents=True, exist_ok=True)

    root_logger: logging.Logger = logging.getLogger()

    timestamp: str = datetime.now(ZoneInfo("Europe/Berlin")).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    run_id: str = uuid4().hex[:8]
    log_path: Path = PATH_LOGS / f"{timestamp}_{run_id}_application.log"

    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            log_path = Path(handler.baseFilename)
            break

    formatter = logging.Formatter(
        LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )

    console_handler: logging.StreamHandler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler: logging.FileHandler = logging.FileHandler(
        log_path,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    listener: logging.handlers.QueueListener = logging.handlers.QueueListener(
        log_queue,
        console_handler,
        file_handler,
        respect_handler_level=True,
    )

    root_logger.setLevel(logging.INFO)
    logging.getLogger("src").setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(logging.handlers.QueueHandler(log_queue))

    listener.start()
    return listener


def setup_worker_logging(
    log_queue: ProcessQueue[logging.LogRecord],
) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    logging.getLogger("src").setLevel(logging.DEBUG)

    root_logger.handlers.clear()
    root_logger.addHandler(logging.handlers.QueueHandler(log_queue))
