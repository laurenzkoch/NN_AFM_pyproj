import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Thread
from unittest.mock import Mock

import pytest

from src.modules.runtime.contracts import (
    PipelineUI,
    PipelineUIFactory,
)
from src.modules.runtime.pipeline import MyPipeline


class PipelineHarness(MyPipeline):
    def create_ui(self) -> PipelineUI:
        return self._initialize_ui(
            title=self.service_name,
        )

    def poll_once(self) -> None:
        self._poll_queue()

    def close_resources(self) -> None:
        self._close_queues()


@contextmanager
def pipeline_with_ui(
    factory: PipelineUIFactory | None,
) -> Iterator[PipelineHarness]:
    pipeline = PipelineHarness(
        service_name="UI test",
        num_kernel=1,
        ui_factory=factory,
    )

    pipeline.max_progress_steps = 125
    pipeline.overall_max_value = 3

    try:
        yield pipeline
    finally:
        pipeline.close_resources()


def test_runtime_pipeline_import_does_not_load_gui() -> None:
    script = """
import sys
from src.modules.runtime.pipeline import MyPipeline

assert MyPipeline.__name__ == "MyPipeline"
assert "tkinter" not in sys.modules
assert not any(
    name == "src.modules.gui"
    or name.startswith("src.modules.gui.")
    for name in sys.modules
)
"""

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_pipeline_uses_factory_and_connects_cancel_callback() -> None:
    ui = Mock(spec=PipelineUI)

    factory = Mock(
        spec=PipelineUIFactory,
        return_value=ui,
    )

    with pipeline_with_ui(factory) as pipeline:
        assert pipeline.create_ui() is ui

        factory.assert_called_once_with(
            max_value=125,
            title="UI test",
            overall_max_value=3,
            on_cancel=pipeline.request_cancel,
        )

        cancel = factory.call_args.kwargs["on_cancel"]

        assert not pipeline.cancellation.is_requested()

        cancel()

        assert pipeline.cancellation.is_requested()


def test_pipeline_without_factory_reports_clear_error() -> None:
    with pipeline_with_ui(None) as pipeline:
        with pytest.raises(
            RuntimeError,
            match="No UI factory configured",
        ):
            pipeline.create_ui()


def test_pipeline_schedules_polling_through_ui_interface() -> None:
    ui = Mock(spec=PipelineUI)
    ui.close_requested = False

    factory = Mock(
        spec=PipelineUIFactory,
        return_value=ui,
    )

    with pipeline_with_ui(factory) as pipeline:
        pipeline.ui = pipeline.create_ui()
        pipeline.service_thread = Thread()

        pipeline.poll_once()

        ui.schedule.assert_called_once()

        delay_ms, callback = ui.schedule.call_args.args

        assert delay_ms == 100
        assert callable(callback)

        callback()

        assert ui.schedule.call_count == 2
