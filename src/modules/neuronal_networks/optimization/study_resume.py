"""Validate a recorded experiment before resuming its study."""

import logging
from pathlib import Path

import optuna

from src.modules.data_managment.experiment_utils import canonical_hash
from src.modules.neuronal_networks.optimization.journal_storage import (
    create_journal_storage,
)

logger = logging.getLogger(__name__)

_RESUME_CONTROL_FILES = frozenset(
    {
        "src/modules/neuronal_networks/optimization/optimizer.py",
        "src/modules/neuronal_networks/optimization/search_worker.py",
        "src/modules/neuronal_networks/optimization/trial_budget.py",
        "src/modules/neuronal_networks/optimization/study_resume.py",
    }
)


def validate_resume_config(
    *,
    saved_config: object,
    saved_fingerprint: object,
    current_config: dict[str, object],
) -> None:
    if not isinstance(saved_config, dict) or not isinstance(saved_fingerprint, str):
        raise ValueError("The study has no valid recorded experiment configuration.")

    if canonical_hash(saved_config) != saved_fingerprint:
        raise ValueError("The recorded experiment fingerprint is inconsistent.")

    saved_settings = {
        key: value for key, value in saved_config.items() if key != "implementation"
    }

    current_settings = {
        key: value for key, value in current_config.items() if key != "implementation"
    }

    if canonical_hash(saved_settings) != canonical_hash(current_settings):
        raise ValueError("Experiment settings differ; this study cannot be resumed.")

    saved_sources = saved_config.get("implementation")
    current_sources = current_config.get("implementation")

    if not isinstance(saved_sources, dict) or not isinstance(current_sources, dict):
        raise ValueError("The experiment implementation hashes are missing.")

    for sources in (saved_sources, current_sources):
        if not sources or any(
            not isinstance(name, str) or not isinstance(digest, str)
            for name, digest in sources.items()
        ):
            raise ValueError("The experiment implementation hashes are invalid.")

    changed_sources = {
        name
        for name in set(saved_sources) | set(current_sources)
        if saved_sources.get(name) != current_sources.get(name)
    }

    incompatible_sources = changed_sources - _RESUME_CONTROL_FILES

    if incompatible_sources:
        raise ValueError(
            "Training implementation differs: "
            + ", ".join(sorted(incompatible_sources))
        )

def find_matching_study(
    *,
    optimization_directory: Path,
    current_config: dict[str, object],
) -> str | None:
    candidates: list[tuple[int, str]] = []

    for journal_path in sorted(
        optimization_directory.glob("*/study.journal")
    ):
        study_name = journal_path.parent.name

        if len(study_name) != 64 or any(
            character not in "0123456789abcdef"
            for character in study_name
        ):
            continue

        study = optuna.load_study(
            study_name=study_name,
            storage=create_journal_storage(journal_path),
        )

        if study.directions != [optuna.study.StudyDirection.MINIMIZE]:
            continue

        recorded_fingerprint = study.user_attrs.get(
            "experiment_fingerprint"
        )

        if recorded_fingerprint != study_name:
            continue

        try:
            validate_resume_config(
                saved_config=study.user_attrs.get("experiment_config"),
                saved_fingerprint=recorded_fingerprint,
                current_config=current_config,
            )
        except ValueError as error:
            logger.debug(
                "Study %s does not match: %s",
                study_name,
                error,
            )
            continue

        completed_count = len(
            study.get_trials(
                deepcopy=False,
                states=(optuna.trial.TrialState.COMPLETE,),
            )
        )

        candidates.append((completed_count, study_name))

    if not candidates:
        return None

    completed_count, selected_name = max(candidates)

    logger.info(
        "Automatically selected study: "
        "study=%s completed=%d matches=%d",
        selected_name,
        completed_count,
        len(candidates),
    )

    return selected_name