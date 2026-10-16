from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    text = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def geometry_ids(geometry: Tensor) -> tuple[str, ...]:
    # Exact identity of the float32 inputs before normalization.
    return tuple(
        canonical_hash([float(x).hex() for x in row])
        for row in geometry.detach().cpu().tolist()
    )


def require_disjoint_split_ids(
    ids: Mapping[str, Collection[str]],
) -> None:
    names = ("train", "validation", "test")
    for i, name in enumerate(names):
        for other in names[i + 1 :]:
            overlap = set(ids[name]) & set(ids[other])
            if overlap:
                raise ValueError(
                    f"Overlapping machines: {name}/{other}, {len(overlap)}"
                )


def validate_search_settings(
    max_layers: int, max_neurons: int, step: int, trials: int, workers: int
) -> None:
    values = (max_layers, max_neurons, step, trials, workers)
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("Search sizes must be positive integers")
    if max_neurons < step:
        raise ValueError("max_neurons must be >= neuron_step")
