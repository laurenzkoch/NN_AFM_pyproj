
import torch
import torch.nn as nn


class MLPGeometryToLatent(nn.Module):

    def __init__(
        self,
        num_input_neurons: int,
        num_output_neurons: int,
        hidden_layer_neurons: tuple[int, ...] = (64, 32, 16)
    ):
        super().__init__()

        layers = []

        in_features = num_input_neurons

        for hidden_size in hidden_layer_neurons:
            layers.append(
                nn.Linear(
                    in_features,
                    hidden_size
                )
            )

            layers.append(
                nn.ReLU()
            )

            in_features = hidden_size

        layers.append(
            nn.Linear(
                in_features,
                num_output_neurons
            )
        )

        self.mlp = nn.Sequential(*layers)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        return self.mlp(x)


