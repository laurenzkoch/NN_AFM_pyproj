from unittest.mock import Mock

import pytest

from src.config.config_mcad_parameters import (
    DEFAULT_VALUES,
    REFERENCE_SPEED_RPM,
)
from src.modules.process.motor_cad_processes import mcad_processing
from src.modules.simulation.reference_run import ReferenceRunResult


def test_base_speed_uses_mechanical_angular_speed() -> None:
    reference = ReferenceRunResult(
        reference_speed_rpm=1000.0,
        back_emf_constant_ll_peak_vs_per_rad=2.0,
        peak_back_emf_phase_v=100.0,
    )

    # 800 V / (2 V*s/rad) = 400 rad/s.
    # Converted to rpm: approximately 3819.718634.
    assert reference.calculate_base_speed_rpm(800.0) == pytest.approx(3819.718634205488)

    assert reference.calculate_base_speed_rpm(400.0) == pytest.approx(1909.859317102744)


@pytest.mark.parametrize(
    "field",
    [
        "reference_speed_rpm",
        "back_emf_constant_ll_peak_vs_per_rad",
        "peak_back_emf_phase_v",
    ],
)
@pytest.mark.parametrize(
    "value",
    [0.0, -1.0, float("nan"), float("inf")],
)
def test_reference_rejects_invalid_results(
    field: str,
    value: float,
) -> None:
    values = {
        "reference_speed_rpm": 1000.0,
        "back_emf_constant_ll_peak_vs_per_rad": 2.0,
        "peak_back_emf_phase_v": 100.0,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        ReferenceRunResult(**values)


@pytest.mark.parametrize(
    "voltage",
    [0.0, -1.0, float("nan"), float("inf")],
)
def test_base_speed_rejects_invalid_voltage(
    voltage: float,
) -> None:
    reference = ReferenceRunResult(1000.0, 2.0, 100.0)

    with pytest.raises(ValueError, match="dc_bus_voltage_v"):
        reference.calculate_base_speed_rpm(voltage)


@pytest.fixture
def reference_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    mcad_processing.MCADProcessor,
    Mock,
    dict[str, float | bool],
    list[dict[str, float | bool]],
]:
    values: dict[str, float | bool] = dict(DEFAULT_VALUES)
    values.update(
        {
            "ShaftSpeed": 2500.0,
            "PeakCurrent": 8.0,
            "RMSCurrentDensity": 6.0,
            "PhaseAdvance": 15.0,
            "OpenCircuitCalculation": False,
            "BackEMFCalculation": False,
            "Ke_Fundamental": 2.0,
            "PeakBackEMFPhase": 100.0,
        }
    )

    calculation_snapshots: list[dict[str, float | bool]] = []

    api = Mock()
    api.get_variable.side_effect = values.__getitem__
    api.set_variable.side_effect = values.__setitem__

    def calculate() -> None:
        calculation_snapshots.append(values.copy())

    api.do_magnetic_calculation.side_effect = calculate

    # Bypass the constructor, which would start a real MCAD instance.
    processor = object.__new__(mcad_processing.MCADProcessor)
    processor.mcad_process = api

    # Geometry validation is outside the scope of these tests.
    monkeypatch.setattr(
        mcad_processing,
        "test_boundary_conditions",
        lambda _names, _values: True,
    )

    return processor, api, values, calculation_snapshots


def test_reference_run_uses_zero_current_and_restores_settings(
    reference_processor,
) -> None:
    processor, api, values, snapshots = reference_processor
    previous_values = values.copy()

    result = processor.execute_reference_run()

    assert result == ReferenceRunResult(
        reference_speed_rpm=REFERENCE_SPEED_RPM,
        back_emf_constant_ll_peak_vs_per_rad=2.0,
        peak_back_emf_phase_v=100.0,
    )

    api.do_magnetic_calculation.assert_called_once_with()
    assert len(snapshots) == 1

    during_calculation = snapshots[0]

    assert during_calculation["ShaftSpeed"] == REFERENCE_SPEED_RPM
    assert during_calculation["PeakCurrent"] == 0.0
    assert during_calculation["RMSCurrentDensity"] == 0.0
    assert during_calculation["PhaseAdvance"] == 0.0
    assert during_calculation["OpenCircuitCalculation"] is True
    assert during_calculation["BackEMFCalculation"] is True

    assert values == previous_values

    queried_names = {call.args[0] for call in api.get_variable.call_args_list}

    assert "Ke_Fundamental" in queried_names
    assert "PeakBackEMFPhase" in queried_names
    assert "SystemEfficiency" not in queried_names


def test_rejected_geometry_does_not_start_calculation(
    reference_processor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, api, values, snapshots = reference_processor
    previous_values = values.copy()

    monkeypatch.setattr(
        mcad_processing,
        "test_boundary_conditions",
        lambda _names, _values: False,
    )

    result = processor.execute_reference_run()

    assert result is None
    api.do_magnetic_calculation.assert_not_called()
    assert snapshots == []
    assert values == previous_values


def test_failed_reference_calculation_restores_settings(
    reference_processor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, _api, values, snapshots = reference_processor
    previous_values = values.copy()

    monkeypatch.setattr(
        processor,
        "do_calculation",
        lambda: False,
    )

    result = processor.execute_reference_run()

    assert result is None
    assert snapshots == []
    assert values == previous_values


def test_invalid_calculated_constant_is_rejected(
    reference_processor,
) -> None:
    processor, _api, values, _snapshots = reference_processor
    values["Ke_Fundamental"] = 0.0

    with pytest.raises(
        ValueError,
        match="back_emf_constant_ll_peak_vs_per_rad",
    ):
        processor.execute_reference_run()
