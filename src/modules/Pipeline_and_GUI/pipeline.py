import logging
import queue
from multiprocessing import Queue
from threading import Thread
from typing import Any
from threading import Lock

from src.config.config_logging import setup_main_logging
from src.modules.Pipeline_and_GUI.events import PipelineEvent, EventType
from src.modules.Pipeline_and_GUI.userinterface import MyUserInterface
from src.modules.neuronal_networks.optimization.optimizer import NNOptimizer
from src.modules.process.resources import determine_number_of_kernels
from src.modules.simulation.simulation import MCADSimulation

logger = logging.getLogger(__name__)


class MyPipeline:

    def __init__(
            self,
            service_name: str,
            num_kernel: int = 1,
    ) -> None:

        self._used = False
        self._start_lock = Lock()

        self.service_name: str = service_name

        self.max_progress_steps: int
        self.num_workers: int = determine_number_of_kernels(num_kernel)

        self.progress_queue = Queue()
        self.log_queue = Queue()
        self.queue_listener = None

        self.ui: MyUserInterface
        self.Service: MCADSimulation | NNOptimizer
        self.service_thread: Thread | None = None
        self.service_exception: Any | None = None

    def run(self) -> None:
        self._claim_run()
        try:
            self.queue_listener = setup_main_logging(self.log_queue)
            self.ui: MyUserInterface = self._initialize_ui(title=self.service_name)
            self.service_thread: Thread = self._initialize_service_thread()
            self._poll_queue()
            self.service_thread.start()
            self.ui.run_ui()
        finally:
            try:
                self._wait_for_thread()
            finally:
                if self.queue_listener is not None:
                    self.queue_listener.stop()

    def _run_service(self) -> None:

        self.progress_queue.put(
            PipelineEvent(
                event_type=EventType.STARTED,
                text=f"{self.service_name} has been started",
            )
        )

        try:
            self.Service.run()
        except Exception as exc:
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
                    text=f"{self.service_name} abgeschlossen",
                )
            )

    def _initialize_service_thread(self) -> Thread:
        return Thread(
            target=self._run_service,
            daemon=False,
        )

    def _initialize_ui(self, title: str) -> MyUserInterface:
        return MyUserInterface(
            max_value=self.max_progress_steps,
            title=title,
        )

    def _wait_for_thread(self) -> None:

        if (
                self.service_thread is not None
                and self.service_thread.ident is not None
        ):
            self.service_thread.join()

        if self.service_exception is not None:
            raise self.service_exception

    def _queue_adapter(self, event: PipelineEvent) -> tuple[int, str, bool]:

        _value = 0
        _text = "default text"
        _true_if_done = False

        return _value, _text, _true_if_done

    def _poll_queue(self) -> None:
        for _ in range(100):
            try:
                event = self.progress_queue.get_nowait()
            except queue.Empty:
                break

            value, text, finished = self._queue_adapter(event)

            self.ui.update_ui(
                text=text,
                _value=value,
            )

            if finished:
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