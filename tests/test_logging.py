from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from logging.handlers import QueueListener
from multiprocessing import get_context
from multiprocessing.queues import Queue as ProcessQueue
from pathlib import Path

import pytest

from src.config import config_logging
from tests.logging_worker import write_markers

logger = logging.getLogger(__name__)


@pytest.mark.subprocess
def test_all_phases_and_worker_use_exactly_one_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "src.config.config_logging.PATH_LOGS",
        tmp_path,
    )
    root = logging.getLogger()
    saved_handlers = root.handlers.copy()
    saved_level = root.level
    project_logger = logging.getLogger("src")
    saved_project_level = project_logger.level
    context = get_context("spawn")
    queue: ProcessQueue[logging.LogRecord] = context.Queue()
    application_handlers: list[logging.Handler] = []
    listener: QueueListener | None = None
    # basicConfig(force=True) darf pytest-eigene Handler nicht schließen.
    root.handlers.clear()
    try:
        config_logging.configure_application_logging()
        application_handlers = root.handlers.copy()
        logger.info("BEFORE_PIPELINE_TEST")
        active_listener: QueueListener = config_logging.setup_main_logging(queue)
        listener = active_listener
        try:
            logger.info("DURING_PIPELINE_TEST")
            with ProcessPoolExecutor(
                max_workers=1,
                mp_context=context,
                initializer=config_logging.setup_worker_logging,
                initargs=(queue,),
            ) as pool:
                worker_pid = pool.submit(write_markers).result(timeout=30)
        finally:
            active_listener.stop()
            root.handlers[:] = application_handlers

            for handler in active_listener.handlers:
                handler.close()

            listener = None
        logger.info("AFTER_PIPELINE_TEST")
        for handler in application_handlers:
            handler.flush()
        files = list(tmp_path.glob("*.log"))
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        for marker in (
            "BEFORE_PIPELINE_TEST",
            "DURING_PIPELINE_TEST",
            "WORKER_INFO_TEST",
            "WORKER_ERROR_TEST",
            "AFTER_PIPELINE_TEST",
            "WORKER_DEBUG_TEST",
        ):
            assert content.count(marker) == 1
        debug_line = next(
            line for line in content.splitlines() if "WORKER_DEBUG_TEST" in line
        )

        assert worker_pid != os.getpid()
        assert f"pid={worker_pid}" in debug_line
        assert "process=" in debug_line
        assert "thread=" in debug_line

        console_output = capsys.readouterr()

        assert "WORKER_DEBUG_TEST" not in console_output.err
        assert "WORKER_INFO_TEST" in console_output.err
        assert not list(tmp_path.glob("*_info.txt"))
        assert not list(tmp_path.glob("*_error.txt"))
    finally:
        if listener is not None:
            listener.stop()
            for handler in listener.handlers:
                handler.close()
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
        project_logger.setLevel(saved_project_level)
        for handler in application_handlers:
            handler.close()
        queue.close()
        queue.join_thread()
