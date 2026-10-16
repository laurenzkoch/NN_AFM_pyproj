from dataclasses import dataclass

import torch
from torch import Tensor

from src.modules.neuronal_networks.manipulating_data import MachineDataTensor
from src.modules.neuronal_networks.model_io import JSONValue
from src.modules.neuronal_networks.surrogate_model import SurrogateModel


@dataclass(frozen=True)
class R2Report:
    mean: float | None
    per_channel: tuple[float | None, ...]


def _checked_pairs(
    y_true: Tensor,
    y_pred: Tensor,
) -> tuple[Tensor, Tensor]:
    if y_true.shape != y_pred.shape:
        raise ValueError("Prediction/reference shapes differ")
    if y_true.ndim < 2 or y_true.numel() == 0:
        raise ValueError("Expected nonempty data with a channel axis")
    if not torch.isfinite(y_true).all():
        raise ValueError("Nonfinite reference")
    if not torch.isfinite(y_pred).all():
        raise ValueError("Nonfinite prediction")
    return (
        y_true.reshape(-1, y_true.shape[-1]).double(),
        y_pred.reshape(-1, y_pred.shape[-1]).double(),
    )


def calc_r_squared_report(y_true: Tensor, y_pred: Tensor) -> R2Report:
    reference, prediction = _checked_pairs(y_true, y_pred)
    residual = ((reference - prediction) ** 2).sum(dim=0)
    total = ((reference - reference.mean(dim=0)) ** 2).sum(dim=0)
    values = tuple(
        None if total[c].item() == 0 else (1 - residual[c] / total[c]).item()
        for c in range(reference.shape[1])
    )
    defined_values: list[float] = [value for value in values if value is not None]

    mean: float | None = (
        sum(defined_values) / len(values)
        if len(defined_values) == len(values)
        else None
    )
    return R2Report(mean=mean, per_channel=values)


def calc_r_squared(y_true: Tensor, y_pred: Tensor) -> float:
    report = calc_r_squared_report(y_true, y_pred)
    if report.mean is None:
        raise ValueError("Overall R2 undefined: constant reference channel")
    return report.mean


def calc_mse(y_true: Tensor, y_pred: Tensor) -> float:
    reference, prediction = _checked_pairs(y_true, y_pred)
    return ((reference - prediction) ** 2).mean().item()


@torch.no_grad()
def evaluate_surrogate(
    model: SurrogateModel,
    data: MachineDataTensor,
    denominator: Tensor,
) -> dict[str, JSONValue]:
    for part in model.submodels:
        part.mlp.eval()
        if part.cnn_autoencoder is not None:
            part.cnn_autoencoder.eval()
    prediction = model.get_prediction(data.geometry_tensor)
    reference, predicted = _checked_pairs(data.map_tensor, prediction)
    scale = denominator.to(reference.device).double()
    errors = (predicted - reference) * scale
    r2 = calc_r_squared_report(reference, predicted)
    return {
        "mse_normalized": ((predicted - reference) ** 2).mean().item(),
        "r2_mean": r2.mean,
        "r2_per_channel": list(r2.per_channel),
        "mae_physical_per_channel": errors.abs().mean(dim=0).tolist(),
        "mse_physical_per_channel": (errors**2).mean(dim=0).tolist(),
        "points_per_channel": reference.shape[0],
    }
