from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from src.modules.data_managment import temp_files
from src.modules.process import resources
from src.modules.simulation.generators.map_generator import map_generator


def test_temporary_copies_have_private_directories_and_unchanged_names(
    tmp_path: Path,
) -> None:
    source = tmp_path / "original.mot"
    source.write_bytes(b"synthetic motor model")
    copies = [
        temp_files.create_temp_file_copy(source, temp_root=tmp_path / "copies")
        for _ in range(2)
    ]
    assert copies[0].parent != copies[1].parent
    for copy in copies:
        assert copy.name == source.name
        assert copy.read_bytes() == source.read_bytes()


def test_failed_copy_removes_only_its_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "original.mot"
    source.write_bytes(b"model")
    parent = tmp_path / "copies"
    parent.mkdir()
    sentinel = parent / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(
        temp_files.shutil, "copy2", Mock(side_effect=OSError("copy failed"))
    )
    with pytest.raises(OSError, match="copy failed"):
        temp_files.create_temp_file_copy(source, temp_root=parent)
    assert list(parent.iterdir()) == [sentinel]


def test_grid_has_correct_axis_order_for_unequal_sizes() -> None:
    grid = map_generator(
        i_min_max=(0.2, 0.8),
        j_min_max=(1.0, 4.0),
        k_min_max=(-3.0, 0.0),
        map_size=(2, 3, 4),
    )

    assert grid.shape == (24, 3)
    assert grid.dtype == np.float64

    assert len(np.unique(grid[:, 0])) == 2
    assert len(np.unique(grid[:, 1])) == 3
    assert len(np.unique(grid[:, 2])) == 4

    np.testing.assert_allclose(
        grid[:5],
        [
            [0.2, 1.0, -3.0],
            [0.2, 1.0, -2.0],
            [0.2, 1.0, -1.0],
            [0.2, 1.0, 0.0],
            [0.2, 2.5, -3.0],
        ],
        rtol=0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        grid[12],
        [0.8, 1.0, -3.0],
        rtol=0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        grid[-1],
        [0.8, 4.0, 0.0],
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("sizes", [(0, 3, 4), (True, 3, 4), (2, -1, 4)])
def test_grid_rejects_invalid_sizes(sizes: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        map_generator(map_size=sizes)


def test_worker_count_is_limited_to_available_cpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 2)
    assert resources.determine_number_of_kernels(4) == 2
    with pytest.raises(TypeError):
        resources.determine_number_of_kernels(True)


@pytest.mark.parametrize(
    "bounds",
    [
        (-0.1, 1.0),
        (1.0, 0.5),
        (0.5, 0.5),
        (float("nan"), 1.0),
        (0.5, float("inf")),
    ],
)
def test_grid_rejects_invalid_speed_bounds(
    bounds: tuple[float, float],
) -> None:
    with pytest.raises(ValueError):
        map_generator(
            i_min_max=bounds,
            map_size=(2, 2, 2),
        )


@pytest.mark.parametrize(
    "bounds",
    [
        (4.0, 1.0),
        (float("nan"), 4.0),
        (1.0, float("inf")),
    ],
)
def test_grid_rejects_invalid_current_density_bounds(
    bounds: tuple[float, float],
) -> None:
    with pytest.raises(ValueError):
        map_generator(
            j_min_max=bounds,
            map_size=(2, 2, 2),
        )

    with pytest.raises(ValueError):
        map_generator(
            k_min_max=bounds,
            map_size=(2, 2, 2),
        )


def test_grid_allows_fixed_single_point_axes() -> None:
    grid = map_generator(
        i_min_max=(0.5, 0.5),
        j_min_max=(3.0, 3.0),
        k_min_max=(-4.0, -4.0),
        map_size=(1, 1, 1),
    )

    np.testing.assert_array_equal(
        grid,
        [[0.5, 3.0, -4.0]],
    )
