from __future__ import annotations

import importlib
import logging
import math
import sys
from collections.abc import Generator
from multiprocessing import get_context
from multiprocessing.queues import Queue as ProcessQueue
from pathlib import Path
from types import ModuleType
from typing import NoReturn

import pandas as pd
import pytest
import torch


def _forbid_external_start(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Automatische Tests dürfen MotorCAD/Tk nicht starten.")


# Nur die externen Grenzen werden ersetzt. Die Projektlogik bleibt echt.
# Auch mit installiertem SDK darf ein Test keine Lizenz/Instanz anfordern.
try:
    sdk = importlib.import_module("ansys.motorcad.core")
except ModuleNotFoundError as exc:
    if exc.name not in {"ansys", "ansys.motorcad", "ansys.motorcad.core"}:
        raise
    ansys = ModuleType("ansys")
    motorcad = ModuleType("ansys.motorcad")
    sdk = ModuleType("ansys.motorcad.core")
    setattr(ansys, "motorcad", motorcad)
    setattr(motorcad, "core", sdk)
    sys.modules.update(
        {
            "ansys": ansys,
            "ansys.motorcad": motorcad,
            "ansys.motorcad.core": sdk,
        }
    )

setattr(sdk, "MotorCAD", _forbid_external_start)
setattr(sdk, "set_motorcad_exe", _forbid_external_start)
if not hasattr(sdk, "MotorCADError"):
    setattr(sdk, "MotorCADError", type("MotorCADError", (Exception,), {}))

try:
    tkinter = importlib.import_module("tkinter")
except ImportError:
    tkinter = ModuleType("tkinter")
    ttk = ModuleType("tkinter.ttk")
    setattr(tkinter, "ttk", ttk)
    sys.modules.update({"tkinter": tkinter, "tkinter.ttk": ttk})
setattr(tkinter, "Tk", _forbid_external_start)

from src.config.config_mcad_parameters import (  # noqa: E402
    AXIS_LIST,
    CSV_DECIMAL_PLACES,
    CSV_PARAMETERS,
    DEFAULT_VALUES,
    MAP_SIZE,
    OUTPUT_NEURON_LIST,
    REFERENCE_SPEED_RPM,
)
from src.modules.data_managment.experiment_utils import canonical_hash  # noqa: E402
from src.modules.data_managment.simulation_metadata import (  # noqa: E402
    SimulationMetadata,
    SimulationPayload,
)
from src.modules.geometry.feature_schema import MACHINE_DEFINITION  # noqa: E402
from src.modules.neuronal_networks.manipulating_data import (  # noqa: E402
    DataLoader,
    MachineDataTensor,
    TrainValidationTestDataTensors,
)
from src.modules.Pipeline_and_GUI.events import PipelineEvent  # noqa: E402


@pytest.fixture
def metadata() -> SimulationMetadata:
    payload: SimulationPayload = {
        "metadata_version": 3,
        "base_model_sha256": "1" * 64,
        "mcad_executable_sha256": "2" * 64,
        "integration_version": "synthetic-test",
        "external_dependency_revision": "test-revision",
        "external_dependency_hashes": {},
        "csv_columns": list(CSV_PARAMETERS),
        "float_decimal_places": CSV_DECIMAL_PLACES,
        "speed_normalization": ("shaft_speed_rpm / base_speed_rpm"),
        "reference_speed_rpm": REFERENCE_SPEED_RPM,
        "reference_current_a": 0.0,
        "back_emf_constant_definition": (
            "fundamental line-line peak / mechanical angular speed; V*s/rad"
        ),
    }

    return {
        "fingerprint": canonical_hash(payload),
        "payload": payload,
    }


@pytest.fixture
def event_queues() -> Generator[
    tuple[ProcessQueue[PipelineEvent], ProcessQueue[logging.LogRecord]]
]:
    context = get_context("spawn")
    progress: ProcessQueue[PipelineEvent] = context.Queue()
    logs: ProcessQueue[logging.LogRecord] = context.Queue()
    try:
        yield progress, logs
    finally:
        # Die kleinen Tests lesen ihre Ereignisse selbst; keine langen Feeder.
        progress.close()
        progress.join_thread()
        logs.close()
        logs.join_thread()


def make_frame(
    machine_count: int = 6,
    first_machine: int = 0,
) -> pd.DataFrame:
    """Create complete synthetic maps with consistent reference data."""
    loader = DataLoader(Path("unused"))
    axes = loader.map_axes

    rows: list[dict[str, float]] = []
    points_per_map = math.prod(MAP_SIZE)

    for machine in range(
        first_machine,
        first_machine + machine_count,
    ):
        dc_bus_voltage_v = 800.0
        reference_ke = 2.0 + machine * 0.01

        base_speed_rpm = dc_bus_voltage_v / reference_ke * 60.0 / math.tau

        for speed_norm in axes[AXIS_LIST[0]]:
            for q_density in axes[AXIS_LIST[1]]:
                for d_density in axes[AXIS_LIST[2]]:
                    current_density = math.hypot(
                        float(q_density),
                        float(d_density),
                    )

                    # Synthetic conductor area: 1 mm².
                    # RMS phase currents therefore numerically equal
                    # the corresponding RMS current densities.
                    peak_current = math.sqrt(2.0) * current_density

                    row = {
                        name: float(DEFAULT_VALUES.get(name, 1.0))
                        for name in CSV_PARAMETERS
                    }

                    row.update(
                        {
                            "Stator_Bore": 70.0 + machine,
                            "Stator_Lam_Dia": 150.0 + 2 * machine,
                            "Magnet_Thickness": 10.0,
                            "DCBusVoltage": dc_bus_voltage_v,
                            "ReferenceSpeedRPM": REFERENCE_SPEED_RPM,
                            "ReferenceKeFundamental": reference_ke,
                            "ReferencePeakBackEMFPhase": (100.0 + machine),
                            "BaseSpeedRPM": base_speed_rpm,
                            "ShaftSpeedNorm": float(speed_norm),
                            "ShaftSpeed": (float(speed_norm) * base_speed_rpm),
                            "PeakCurrent": peak_current,
                            "RMSCurrentDensity": current_density,
                            "RMSPhaseCurrent_D": float(d_density),
                            "RMSPhaseCurrent_Q": float(q_density),
                            "PhaseAdvance": math.degrees(
                                math.atan2(
                                    -float(d_density),
                                    float(q_density),
                                )
                            ),
                        }
                    )

                    point_index = len(rows) % points_per_map

                    for channel, name in enumerate(OUTPUT_NEURON_LIST):
                        row[name] = machine * 100.0 + point_index + channel

                    rows.append(row)

    return pd.DataFrame(rows, columns=CSV_PARAMETERS)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, sep=";", decimal=",", index=False)


@pytest.fixture
def tensor_data() -> TrainValidationTestDataTensors:
    generator = torch.Generator().manual_seed(123)
    geometry = torch.rand((6, len(MACHINE_DEFINITION)), generator=generator)
    maps = torch.rand((6, *MAP_SIZE, len(OUTPUT_NEURON_LIST)), generator=generator)
    return TrainValidationTestDataTensors(
        train=MachineDataTensor(geometry[:5].clone(), maps[:5].clone()),
        validation=MachineDataTensor(geometry[5:].clone(), maps[5:].clone()),
        test=None,
        map_axes={
            name: axis.copy()
            for name, axis in DataLoader(Path("unused")).map_axes.items()
        },
        target_minimum=torch.zeros(len(OUTPUT_NEURON_LIST)),
        target_denominator=torch.ones(len(OUTPUT_NEURON_LIST)),
        split_machine_ids={
            "train": tuple(f"train-{i}" for i in range(5)),
            "validation": ("validation-0",),
            "test": (),
        },
        source_hashes={"synthetic.csv": "a" * 64},
    )
