import logging
import logging.handlers
from datetime import datetime
from multiprocessing import Queue
from zoneinfo import ZoneInfo

from src.config.config_paths import PATH_LOGS


def setup_main_logging(log_queue: Queue):
    PATH_LOGS.mkdir(
        parents=True,
        exist_ok=True,
    )

    current_time = datetime.now(
        ZoneInfo("Europe/Berlin")
    ).strftime("%Y.%m.%d_%H-%M-%S")

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    info_file_handler = logging.FileHandler(
        PATH_LOGS / f"{current_time}_info.txt",
        mode="w",
        encoding="utf-8",
    )
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(formatter)

    error_file_handler = logging.FileHandler(
        PATH_LOGS / f"{current_time}_error.txt",
        mode="w",
        encoding="utf-8",
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)

    listener = logging.handlers.QueueListener(
        log_queue,
        console_handler,
        info_file_handler,
        error_file_handler,
        respect_handler_level=True,
    )

    # Logging des Hauptprozesses ebenfalls über die Queue
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(
        logging.handlers.QueueHandler(log_queue)
    )

    listener.start()

    return listener


def setup_worker_logging(log_queue: Queue):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    root_logger.handlers.clear()

    root_logger.addHandler(
        logging.handlers.QueueHandler(log_queue)
    )
