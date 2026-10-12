import torch


def calc_r_squared(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("Prediction and reference shapes must match")

    # [Maschinen, D, H, W, Kanäle] -> [alle Punkte, Kanäle]
    reference = y_true.reshape(-1, y_true.shape[-1])
    prediction = y_pred.reshape(-1, y_pred.shape[-1])

    channel_mean = reference.mean(dim=0)

    ss_res = ((reference - prediction) ** 2).sum(dim=0)
    ss_tot = ((reference - channel_mean) ** 2).sum(dim=0)

    if (ss_tot == 0).any():
        raise ValueError(
            "R² is undefined for a constant reference channel"
        )

    r2_per_channel = 1 - ss_res / ss_tot

    return r2_per_channel.mean().item()


def calc_mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    mse = torch.nn.MSELoss()(y_pred, y_true)
    return mse.item()
