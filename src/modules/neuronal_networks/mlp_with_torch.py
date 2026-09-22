from dataclasses import dataclass

import torch
import torch.nn as nn

class MLP(nn.Module):

    """
    Fully connected multilayer perceptron for regression.

    Parameters
    ----------
    num_input_neurons:
        Number of input features.

    num_output_neurons:
        Number of predicted output values.

    hidden_layer_neurons:
        Number of neurons in each hidden layer.
    """

    def __init__(
        self,
        num_input_neurons: int,
        num_output_neurons: int,
        hidden_layer_neurons: tuple[int, ...] = (64, 32, 16),
        activation: type[nn.Module] = nn.ReLU
    ):
        super().__init__()

        if num_input_neurons <= 0:
            raise ValueError(
                "num_input_neurons must be greater than 0."
            )

        if num_output_neurons <= 0:
            raise ValueError(
                "num_output_neurons must be greater than 0."
            )

        if any(n <= 0 for n in hidden_layer_neurons):
            raise ValueError(
                "All hidden layer sizes must be greater than 0."
            )


        layers = []

        in_features = num_input_neurons

        for hidden_size in hidden_layer_neurons:
            layers.append(
                nn.Linear(
                    in_features,
                    hidden_size
                )
            )

            layers.append(activation())

            in_features = hidden_size

        layers.append(
            nn.Linear(
                in_features,
                num_output_neurons
            )
        )

        self.network = nn.Sequential(*layers)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        return self.network(x)

@dataclass
class TrainedMLP:
    model_state_dict: dict
    num_input_neurons: int
    num_output_neurons: int
    hidden_layer_neurons: tuple[int, ...]
    activation: type[nn.Module]

