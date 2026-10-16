from __future__ import annotations

import logging
import logging.handlers
from datetime import datetime
from multiprocessing.queues import Queue as ProcessQueue
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from src.config.config_paths import PATH_LOGS


def configure_application_logging() -> None:
    PATH_LOGS.mkdir(parents=True, exist_ok=True)

    timestamp: str = datetime.now(ZoneInfo("Europe/Berlin")).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    run_id: str = uuid4().hex[:8]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                PATH_LOGS / f"{timestamp}_{run_id}_application.log",
                encoding="utf-8",
            ),
        ],
        force=True,
    )


def setup_main_logging(
    log_queue: ProcessQueue[logging.LogRecord],
) -> logging.handlers.QueueListener:
    PATH_LOGS.mkdir(parents=True, exist_ok=True)

    root_logger: logging.Logger = logging.getLogger()

    # Die bereits durch configure_application_logging()
    # eingerichtete Anwendungsdatei weiterverwenden.
    log_path: Path | None = None

    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            log_path = Path(handler.baseFilename)
            break

    if log_path is None:
        timestamp: str = datetime.now(ZoneInfo("Europe/Berlin")).strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        run_id: str = uuid4().hex[:8]
        log_path = PATH_LOGS / f"{timestamp}_{run_id}_application.log"

    formatter: logging.Formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler: logging.StreamHandler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler: logging.FileHandler = logging.FileHandler(
        log_path,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    listener: logging.handlers.QueueListener = logging.handlers.QueueListener(
        log_queue,
        console_handler,
        file_handler,
        respect_handler_level=True,
    )

    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(logging.handlers.QueueHandler(log_queue))

    listener.start()
    return listener


def setup_worker_logging(
    log_queue: ProcessQueue[logging.LogRecord],
) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    root_logger.handlers.clear()

    root_logger.addHandler(logging.handlers.QueueHandler(log_queue))
