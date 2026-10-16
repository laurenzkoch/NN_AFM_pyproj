import math

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        num_input_neurons: int,
        output_dims: tuple[int, ...],
        hidden_layer_neurons: tuple[int, ...],
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        if num_input_neurons <= 0:
            raise ValueError("num_input_neurons must be greater than 0.")

        if not output_dims or any(d <= 0 for d in output_dims):
            raise ValueError("output_dims must be positive")

        if any(n <= 0 for n in hidden_layer_neurons):
            raise ValueError("All hidden layer sizes must be greater than 0.")

        self.output_dims = output_dims
        self.model_state_dict: dict[str, torch.Tensor]

        layers: list[nn.Module] = []

        in_features = num_input_neurons

        for hidden_size in hidden_layer_neurons:
            layers.append(nn.Linear(in_features, hidden_size))

            layers.append(activation())

            in_features = hidden_size

        layers.append(nn.Linear(in_features, math.prod(output_dims)))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prediction = self.network(x)
        prediction = prediction.reshape(x.shape[0], *self.output_dims)
        return prediction
