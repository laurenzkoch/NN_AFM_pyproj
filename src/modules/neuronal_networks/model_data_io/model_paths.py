from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from src.config.config_paths import PATH_DATA_TRAINED_NN

type ModelCategory = Literal[
    "Reference_model",
    "Combi_model",
    "AutoencoderCollections",
]

MODEL_FILENAMES: dict[ModelCategory, str] = {
    "Reference_model": "SurrogateModel.pt",
    "Combi_model": "SurrogateModel.pt",
    "AutoencoderCollections": "AutoencoderCollection.pt",
}


def make_model_path(category: ModelCategory) -> Path:
    """Determine a unique destination for one complete trained model."""
    if category not in MODEL_FILENAMES:
        raise ValueError(f"Unknown model category: {category}")

    timestamp: str = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_name: str = f"{timestamp}_{uuid4().hex[:8]}"

    return Path(PATH_DATA_TRAINED_NN) / category / run_name / MODEL_FILENAMES[category]


def resolve_model_directory(model_folder: str | Path) -> Path:
    """Resolve an existing model directory within the trained-model root."""
    root: Path = Path(PATH_DATA_TRAINED_NN).resolve()
    supplied_path: Path = Path(model_folder)

    directory: Path = (
        supplied_path if supplied_path.is_absolute() else root / supplied_path
    ).resolve()

    if directory == root or not directory.is_relative_to(root):
        raise ValueError("Specify a model directory inside PATH_DATA_TRAINED_NN.")

    if not directory.is_dir():
        raise NotADirectoryError(f"Model directory not found: {directory}")

    return directory
