"""Distribute the remaining successful-trial budget across workers."""


def remaining_trial_counts(
    *,
    target_trials: int,
    completed_trials: int,
    num_workers: int,
) -> tuple[int, ...]:
    for name, value, minimum in (
        ("target_trials", target_trials, 0),
        ("completed_trials", completed_trials, 0),
        ("num_workers", num_workers, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(
                f"{name} must be an integer greater than or equal to {minimum}."
            )

    remaining = max(
        0,
        target_trials - completed_trials,
    )

    base_count, extra = divmod(
        remaining,
        num_workers,
    )

    return tuple(
        base_count + (1 if index < extra else 0) for index in range(num_workers)
    )
