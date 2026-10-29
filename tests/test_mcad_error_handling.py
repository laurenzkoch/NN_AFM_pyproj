import logging
import pickle
from unittest.mock import Mock

import pytest

from src.modules.process.motor_cad_processes import (
    mcad_errors as errors,
)
from src.modules.process.motor_cad_processes import (
    mcad_instance_handling as instances,
)
from src.modules.process.motor_cad_processes import (
    mcad_processing as processing,
)
from src.modules.simulation import simulation as simulations


@pytest.fixture
def processor() -> processing.MCADProcessor:
    instance = object.__new__(processing.MCADProcessor)
    instance.mcad_process = Mock()
    instance._close_attempted = False
    return instance


@pytest.mark.parametrize(
    "exception_type",
    [
        errors.MCADExecutionError,
        errors.MCADCommunicationError,
        errors.MCADCalculationError,
        errors.MCADInitializationError,
        errors.MCADResultError,
        errors.MCADCleanupError,
    ],
)
def test_exceptions_can_cross_process_boundaries(
    exception_type: type[errors.MCADExecutionError],
) -> None:
    original = exception_type("Test failure")

    restored = pickle.loads(pickle.dumps(original))

    assert type(restored) is exception_type
    assert restored.args == original.args


def test_parameter_communication_error_preserves_cause(
    processor: processing.MCADProcessor,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(
        logging.DEBUG,
        logger=processing.__name__,
    )

    original = processing.pymotorcad.MotorCADError(
        "RPC Communication failed: connection aborted"
    )
    processor.mcad_process.set_variable.side_effect = original

    with pytest.raises(
        errors.MCADCommunicationError
    ) as captured:
        processor.set_value_in_mcad(
            "Tooth_Tip_Depth",
            1.8,
        )

    assert captured.value.__cause__ is original
    assert "Tooth_Tip_Depth" in str(captured.value)
    assert "1.8" in str(captured.value)

    assert "MCAD operation failed" in caplog.text
    assert "elapsed_s=" in caplog.text


def test_result_communication_error_preserves_cause(
    processor: processing.MCADProcessor,
) -> None:
    original = processing.pymotorcad.MotorCADError(
        "RPC Communication failed: remote disconnected"
    )
    processor.mcad_process.get_variable.side_effect = original

    with pytest.raises(
        errors.MCADCommunicationError
    ) as captured:
        processor.get_value_from_mcad("Ke_Fundamental")

    assert captured.value.__cause__ is original


@pytest.mark.parametrize(
    "raw_value",
    [
        None,
        [],
        "not a number",
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_numeric_result_is_rejected(
    processor: processing.MCADProcessor,
    raw_value: object,
) -> None:
    processor.mcad_process.get_variable.return_value = raw_value

    with pytest.raises(errors.MCADResultError):
        processor.get_value_from_mcad("Ke_Fundamental")


@pytest.mark.parametrize(
    ("message", "expected_type"),
    [
        (
            "Failed to Solve Problem",
            errors.MCADCalculationError,
        ),
        (
            "RPC Communication failed: connection aborted",
            errors.MCADCommunicationError,
        ),
    ],
)
def test_calculation_errors_are_not_converted_to_false(
    processor: processing.MCADProcessor,
    message: str,
    expected_type: type[errors.MCADExecutionError],
) -> None:
    original = processing.pymotorcad.MotorCADError(message)
    processor.mcad_process.do_magnetic_calculation.side_effect = (
        original
    )

    with pytest.raises(expected_type) as captured:
        processor.do_calculation()

    assert captured.value.__cause__ is original
    processor.mcad_process.do_magnetic_calculation.assert_called_once()


def test_close_attempt_is_not_repeated(
    processor: processing.MCADProcessor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.modules.process.motor_cad_processes."
        "mcad_processing.time.sleep",
        lambda _seconds: None,
    )

    processor.close()
    processor.close()

    processor.mcad_process.quit.assert_called_once()


def test_shutdown_discards_instance_even_when_close_fails(
    processor: processing.MCADProcessor,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        "src.modules.process.motor_cad_processes."
        "mcad_instance_handling._mcad_instance",
        processor,
    )

    close = Mock(
        side_effect=errors.MCADCleanupError("quit failed")
    )
    monkeypatch.setattr(processor, "close", close)

    instances.shutdown_mcad_instance()

    close.assert_called_once_with()
    assert "cleanup failed" in caplog.text

    with pytest.raises(
        RuntimeError,
        match="not initialized",
    ):
        instances.get_mcad_instance()


def test_strict_shutdown_reports_cleanup_failure(
    processor: processing.MCADProcessor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.modules.process.motor_cad_processes."
        "mcad_instance_handling._mcad_instance",
        processor,
    )

    original = errors.MCADCleanupError("quit failed")

    monkeypatch.setattr(
        processor,
        "close",
        Mock(side_effect=original),
    )

    with pytest.raises(errors.MCADCleanupError) as captured:
        instances.shutdown_mcad_instance(
            raise_on_error=True,
        )

    assert captured.value.__cause__ is original


def test_worker_cleanup_failure_preserves_original_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    processor = Mock()
    original = errors.MCADCommunicationError(
        "connection lost"
    )
    processor.execute_mcad_simulation.side_effect = original

    monkeypatch.setattr(
        "src.modules.simulation.simulation."
        "mcad_instance_handling.get_mcad_instance",
        lambda: processor,
    )

    cleanup = Mock(
        side_effect=errors.MCADCleanupError("quit failed")
    )
    monkeypatch.setattr(
        "src.modules.simulation.simulation."
        "mcad_instance_handling.shutdown_mcad_instance",
        cleanup,
    )

    with pytest.raises(
        errors.MCADCommunicationError
    ) as captured:
        simulations._execute_simulation(
            {},
            log_context="batch=b1 geometry=g1 task=t1",
        )

    assert captured.value is original
    cleanup.assert_called_once_with()

    assert "batch=b1 geometry=g1 task=t1" in caplog.text
    assert "phase=load_calculation" in caplog.text
    assert "Unexpected worker cleanup error" in caplog.text


def test_pool_cleanup_failure_does_not_escape(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pool = Mock()
    pool.kill_workers.side_effect = RuntimeError(
        "termination failed"
    )

    simulations.MCADSimulation._stop_pool_after_error(
        pool,
        "batch=b1",
    )

    pool.kill_workers.assert_called_once_with()
    assert "Process pool termination failed" in caplog.text
    assert "batch=b1" in caplog.text

def test_instance_creation_failure_is_logged_and_propagated(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    original = errors.MCADInitializationError(
        "instance creation failed"
    )

    monkeypatch.setattr(
        "src.modules.process.motor_cad_processes."
        "mcad_instance_handling._mcad_instance",
        None,
    )
    monkeypatch.setattr(
        "src.modules.process.motor_cad_processes."
        "mcad_instance_handling.setup_worker_logging",
        Mock(),
    )
    monkeypatch.setattr(
        "src.modules.process.motor_cad_processes."
        "mcad_instance_handling.protect_worker_process_tree",
        Mock(),
    )
    monkeypatch.setattr(
        "src.modules.process.motor_cad_processes."
        "mcad_instance_handling.Finalize",
        Mock(),
    )
    monkeypatch.setattr(
        "src.modules.process.motor_cad_processes."
        "mcad_instance_handling.MCADProcessor",
        Mock(side_effect=original),
    )

    with pytest.raises(
        errors.MCADInitializationError
    ) as captured:
        instances.init_mcad_instance(Mock())

    assert captured.value is original
    assert "phase=instance_creation" in caplog.text