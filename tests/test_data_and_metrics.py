from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.config.config_mcad_parameters import OUTPUT_NEURON_LIST
from src.modules.data_managment.experiment_utils import (
    file_sha256,
    require_disjoint_split_ids,
)
from src.modules.neuronal_networks.manipulating_data import DataLoader
from src.modules.neuronal_networks.testing import calc_mse, calc_r_squared_report
from tests.conftest import make_frame, write_frame


def test_seven_channel_metrics_and_constant_reference() -> None:
    reference = torch.arange(4, dtype=torch.float64).reshape(4, 1).repeat(1, 7)
    assert calc_mse(reference, reference) == 0
    assert calc_r_squared_report(reference, reference).mean == 1
    prediction = reference + torch.arange(7, dtype=torch.float64) / 10
    assert calc_mse(reference, prediction) == pytest.approx(0.13)
    report = calc_r_squared_report(reference, prediction)
    assert report.mean == pytest.approx(0.896)
    assert len(report.per_channel) == 7
    reference[:, 0] = 0
    assert calc_r_squared_report(reference, prediction).mean is None


@pytest.mark.parametrize("problem", ["shape", "nan", "inf", "empty"])
def test_metrics_reject_invalid_inputs(problem: str) -> None:
    reference = torch.ones((4, 7))
    prediction = reference.clone()
    if problem == "shape":
        prediction = prediction[:, :1]
    elif problem == "nan":
        prediction[0, 0] = float("nan")
    elif problem == "inf":
        reference[0, 0] = float("inf")
    else:
        reference = reference[:0]
        prediction = prediction[:0]
    with pytest.raises(ValueError):
        calc_mse(reference, prediction)


def test_loader_normalizes_using_training_only_and_preserves_sources(
    tmp_path: Path,
) -> None:
    frame = make_frame()
    _, validation_indices = DataLoader._split_indices(6, 0.2)
    validation_id = int(validation_indices[0])
    frame.loc[frame["Stator_Bore"] == 70 + validation_id, OUTPUT_NEURON_LIST] = (
        1_000_000.0
    )
    external = make_frame(1, 20)
    external.loc[:, OUTPUT_NEURON_LIST] = 2_000_000.0
    main_path = tmp_path / "main.csv"
    test_path = tmp_path / "test.csv"
    write_frame(main_path, frame)
    write_frame(test_path, external)
    before = {path: file_sha256(path) for path in (main_path, test_path)}
    data = DataLoader(tmp_path).load("main.csv", ("test.csv",))
    assert data.test is not None
    assert data.target_denominator.max().item() < 1_000_000
    assert data.validation.map_tensor.min().item() > 1
    assert data.test.map_tensor.min().item() > 1
    for split in (data.train, data.validation, data.test):
        assert torch.isfinite(split.geometry_tensor).all()
        assert torch.isfinite(split.map_tensor).all()
    require_disjoint_split_ids(data.split_machine_ids)
    assert data.source_hashes == {
        "main.csv": before[main_path],
        "test.csv": before[test_path],
    }
    assert all(file_sha256(path) == digest for path, digest in before.items())


def test_loader_rejects_overlapping_external_test_machines(tmp_path: Path) -> None:
    frame = make_frame()
    write_frame(tmp_path / "main.csv", frame)
    write_frame(tmp_path / "overlap.csv", frame.copy())
    with pytest.raises(ValueError, match="Overlapping machines"):
        DataLoader(tmp_path).load("main.csv", ("overlap.csv",))


@pytest.mark.parametrize(
    "column",
    [
        "RMSCurrentDensity",
        "PeakCurrent",
        "Magnet_Length",
        "Slot_Depth",
        "DCBusVoltage",
        "ReferenceSpeedRPM",
        "ReferenceKeFundamental",
        "ReferencePeakBackEMFPhase",
        "BaseSpeedRPM",
    ],
)
def test_zero_denominators_are_rejected(column: str) -> None:
    frame = make_frame(1)
    frame.loc[0, column] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        DataLoader._recalculate_data(frame)


def test_incomplete_machine_is_removed_as_a_whole(tmp_path: Path) -> None:
    frame = make_frame()
    write_frame(tmp_path / "main.csv", frame.drop(index=0))
    data = DataLoader(tmp_path).load("main.csv")
    assert (
        len(data.split_machine_ids["train"]) + len(data.split_machine_ids["validation"])
        == 5
    )


def test_duplicate_operating_point_invalidates_its_machine(tmp_path: Path) -> None:
    import pandas as pd

    frame = make_frame()
    write_frame(
        tmp_path / "main.csv", pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    )
    data = DataLoader(tmp_path).load("main.csv")
    assert (
        len(data.split_machine_ids["train"]) + len(data.split_machine_ids["validation"])
        == 5
    )
