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
        i_min_max=(100, 200), j_min_max=(1, 4), k_min_max=(-3, 0), map_size=(2, 3, 4)
    )
    assert grid.shape == (24, 3)
    assert len(np.unique(grid[:, 0])) == 2
    assert len(np.unique(grid[:, 1])) == 3  # Spalte q
    assert len(np.unique(grid[:, 2])) == 4  # Spalte d


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
