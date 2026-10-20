from __future__ import annotations

import logging
import queue
from logging.handlers import QueueListener
from multiprocessing import Queue
from multiprocessing.queues import Queue as ProcessQueue
from threading import Lock, Thread
from typing import Protocol

from src.config.config_logging import setup_main_logging
from src.modules.Pipeline_and_GUI.cancellation import (
    CancellationToken,
    PipelineCancelledError,
)
from src.modules.Pipeline_and_GUI.events import EventType, PipelineEvent
from src.modules.Pipeline_and_GUI.userinterface import MyUserInterface
from src.modules.process.resources import determine_number_of_kernels

logger = logging.getLogger(__name__)


class PipelineService(Protocol):
    def run(self) -> None: ...


class MyPipeline:
    def __init__(
        self,
        service_name: str,
        num_kernel: int = 1,
    ) -> None:
        self._used = False
        self._start_lock = Lock()
        self.cancellation: CancellationToken = CancellationToken()

        self.service_name: str = service_name

        self.max_progress_steps: int
        self.overall_max_value: int | None = None
        self.overall_value: int = 0
        self.num_workers: int = determine_number_of_kernels(num_kernel)

        self.progress_queue: ProcessQueue[PipelineEvent] = Queue()
        self.log_queue: ProcessQueue[logging.LogRecord] = Queue()
        self._queues_closed: bool = False
        self.queue_listener: QueueListener | None = None

        self.ui: MyUserInterface | None = None
        self._terminal_received = False

        self.service: PipelineService
        self.service_thread: Thread | None = None
        self.service_exception: BaseException | None = None

    def _close_queues(self) -> None:
        if self._queues_closed:
            return

        self.progress_queue.cancel_join_thread()
        self.progress_queue.close()

        self.log_queue.close()
        self.log_queue.join_thread()

        self._queues_closed = True

    def _run_pipeline(self) -> None:
        root_logger = logging.getLogger()
        saved_handlers = list(root_logger.handlers)
        saved_level = root_logger.level
        try:
            self.queue_listener = setup_main_logging(self.log_queue)
            self.ui = self._initialize_ui(title=self.service_name)
            self.service_thread = self._initialize_service_thread()
            self.service_thread.start()
            self._poll_queue()
            self.ui.run_ui()
        finally:
            try:
                self._wait_for_thread()
            finally:
                try:
                    if self.ui is not None:
                        self.ui.destroy()
                finally:
                    try:
                        if self.queue_listener is not None:
                            self.queue_listener.stop()
                            for handler in self.queue_listener.handlers:
                                handler.close()
                    finally:
                        root_logger.handlers[:] = saved_handlers
                        root_logger.setLevel(saved_level)
                        self._close_queues()

    def _run_service(self) -> None:
        self.progress_queue.put(
            PipelineEvent(
                event_type=EventType.STARTED,
                text=f"{self.service_name} has been started",
            )
        )

        try:
            self.cancellation.raise_if_requested()
            self.service.run()
            self.cancellation.raise_if_requested()
        except PipelineCancelledError as exc:
            self.service_exception = exc
            logger.info("Pipeline cancelled: %s", self.service_name)
            self.progress_queue.put(
                PipelineEvent(
                    event_type=EventType.CANCELLED,
                    text=f"{self.service_name} cancelled",
                )
            )
        except BaseException as exc:
            self.service_exception = exc
            self.progress_queue.put(
                PipelineEvent(
                    event_type=EventType.FAILED,
                    text=f"{type(exc).__name__}: {exc}",
                )
            )
            logger.exception(exc)
        else:
            self.progress_queue.put(
                PipelineEvent(
                    event_type=EventType.FINISHED,
                    text=f"{self.service_name} completed",
                )
            )

    def _initialize_service_thread(self) -> Thread:
        return Thread(
            target=self._run_service,
            daemon=False,
        )

    def request_cancel(self) -> None:
        if self.cancellation.is_requested():
            return

        self.cancellation.request()
        logger.info("Cancellation requested for pipeline: %s", self.service_name)

    def _initialize_ui(self, title: str) -> MyUserInterface:
        return MyUserInterface(
            max_value=self.max_progress_steps,
            title=title,
            overall_max_value=self.overall_max_value,
            on_cancel=self.request_cancel,
        )

    def _wait_for_thread(self) -> None:
        thread = self.service_thread
        if thread is not None and thread.ident is not None:
            while thread.is_alive():
                thread.join(timeout=0.1)
                # Defensive drain if mainloop exits unexpectedly.
                for _ in range(1000):
                    try:
                        self.progress_queue.get_nowait()
                    except queue.Empty:
                        break
        if self.service_exception is not None:
            raise self.service_exception

    def _queue_adapter(self, event: PipelineEvent) -> tuple[int, str, bool]:
        _value = 0
        _text = "default text"
        _true_if_done = False

        return _value, _text, _true_if_done

    def _poll_queue(self) -> None:
        ui = self.ui
        thread = self.service_thread

        if ui is None or thread is None:
            raise RuntimeError("GUI und Servicethread are not initialized.")

        saw_empty = False
        for _ in range(100):
            try:
                event = self.progress_queue.get_nowait()
            except queue.Empty:
                saw_empty = True
                break

            value, text, terminal = self._queue_adapter(event)

            if not ui.close_requested:
                ui.update_ui(
                    text=text,
                    _value=value,
                    max_steps=self.max_progress_steps,
                    overall_value=self.overall_value,
                )

            self._terminal_received = self._terminal_received or terminal

        if self._terminal_received and not thread.is_alive() and saw_empty:
            ui.mark_finished()
            return

        self.ui.root.after(100, self._poll_queue)

    def _claim_run(self) -> None:
        with self._start_lock:
            if self._used:
                raise RuntimeError(
                    "This pipeline instance has already been used. "
                    "Create a new instance for another run."
                )

            self._used = True
