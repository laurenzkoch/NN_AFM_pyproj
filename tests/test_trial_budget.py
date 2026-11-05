import pytest

from src.modules.neuronal_networks.optimization.trial_budget import (
    remaining_trial_counts,
)


@pytest.mark.parametrize(
    ("completed", "expected"),
    [
        (0, (100, 100, 100, 100)),
        (350, (13, 13, 12, 12)),
        (398, (1, 1, 0, 0)),
        (400, (0, 0, 0, 0)),
        (450, (0, 0, 0, 0)),
    ],
)
def test_only_remaining_trials_are_distributed(
    completed: int,
    expected: tuple[int, ...],
) -> None:
    result = remaining_trial_counts(
        target_trials=400,
        completed_trials=completed,
        num_workers=4,
    )

    assert result == expected
    assert sum(result) == max(
        0,
        400 - completed,
    )
