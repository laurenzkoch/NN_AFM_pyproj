from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.config.config_mcad_parameters import CSV_PARAMETERS
from src.modules.data_managment.csv_io import import_data_from_csv
from src.modules.data_managment.experiment_utils import canonical_hash
from src.modules.data_managment.simulation_metadata import (
    SimulationMetadata,
    metadata_path,
    write_metadata,
)
from src.modules.simulation.result_cache import ResultCache
from tests.conftest import make_frame, write_frame


def test_cache_prefers_destination_after_append(
    tmp_path: Path,
    metadata: SimulationMetadata,
) -> None:
    source = tmp_path / "source.csv"
    destination = tmp_path / "destination.csv"

    frame = make_frame(1).iloc[:1].copy()
    write_frame(source, frame)

    source_before = source.read_bytes()

    cache = ResultCache(
        csv_path=destination,
        expected_metadata=metadata,
        cache_directory=tmp_path,
    )

    parameters = {
        "Stator_Bore": float(frame.iloc[0]["Stator_Bore"]),
        "ShaftSpeedNorm": float(frame.iloc[0]["ShaftSpeedNorm"]),
    }

    assert cache.find_known_result(parameters) == (
        source.resolve().as_posix(),
        0,
    )

    cache.prepare_output()

    cache.append_result(
        cache.get_row(
            source.resolve().as_posix(),
            0,
        )
    )

    assert cache.find_known_result(parameters) == (
        destination.resolve().as_posix(),
        0,
    )

    assert len(cache.dataframes[destination.resolve().as_posix()]) == 1

    stored = import_data_from_csv(destination)

    assert len(stored) == 1

    np.testing.assert_allclose(
        stored.iloc[0][CSV_PARAMETERS].to_numpy(dtype=float),
        cache.get_row(source.resolve().as_posix(), 0),
        rtol=1e-10,
        atol=1e-9,
    )

    assert source.read_bytes() == source_before


@pytest.mark.parametrize(
    "problem",
    [
        "metadata",
        "missing_metadata",
        "schema",
        "nonfinite",
    ],
)
def test_cache_ignores_invalid_sources(
    tmp_path: Path,
    metadata: SimulationMetadata,
    problem: str,
) -> None:
    source = tmp_path / "bad.csv"
    frame = make_frame(1).iloc[:1].copy()

    if problem == "schema":
        frame = frame.drop(columns=CSV_PARAMETERS[-1])
    elif problem == "nonfinite":
        frame.loc[
            frame.index[0],
            "ShaftTorque",
        ] = float("inf")

    write_frame(source, frame)

    if problem == "metadata":
        changed: SimulationMetadata = deepcopy(metadata)

        changed["payload"]["base_model_sha256"] = "9" * 64
        changed["fingerprint"] = canonical_hash(changed["payload"])

        write_metadata(source, changed)

    elif problem == "missing_metadata":
        metadata_path(source).unlink()

    cache = ResultCache(
        csv_path=tmp_path / "destination.csv",
        expected_metadata=metadata,
        cache_directory=tmp_path,
    )

    assert cache.dataframes == {}
