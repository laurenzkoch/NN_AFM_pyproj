from copy import deepcopy

import pytest

from src.modules.data_managment.experiment_utils import canonical_hash
from src.modules.neuronal_networks.optimization.study_resume import (
    validate_resume_config,
)


@pytest.mark.parametrize(
    "change",
    ["control", "workers", "training", "objective"],
)
def test_resume_configuration_rules(change: str) -> None:
    prefix = "src/modules/neuronal_networks/optimization/"

    saved: dict[str, object] = {
        "workers": 4,
        "training": {"epochs": 2000},
        "implementation": {
            prefix + "optimizer.py": "1" * 64,
            prefix + "objective.py": "2" * 64,
        },
    }

    current: dict[str, object] = deepcopy(saved)

    sources = current["implementation"]
    assert isinstance(sources, dict)

    if change == "control":
        sources[prefix + "optimizer.py"] = "9" * 64
    elif change == "workers":
        current["workers"] = 2
    elif change == "training":
        current["training"] = {"epochs": 1000}
    else:
        sources[prefix + "objective.py"] = "9" * 64

    if change == "control":
        validate_resume_config(
            saved_config=saved,
            saved_fingerprint=canonical_hash(saved),
            current_config=current,
        )
    else:
        with pytest.raises(ValueError):
            validate_resume_config(
                saved_config=saved,
                saved_fingerprint=canonical_hash(saved),
                current_config=current,
            )
