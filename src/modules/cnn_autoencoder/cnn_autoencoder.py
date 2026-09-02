
import torch.nn as nn

class Conv3dAutoencoder(nn.Module):

    def __init__(
            self,
            latent_dim: int = 16,
            num_output_neurons: int = 7,
            points_per_dim: int = 10,
            kernel_size: int = 3,
            out_channels: int = 15
    ):
        super().__init__()

        self.dim_of_map = 3
        self.points_per_dim = points_per_dim
        self.out_channels = out_channels

        self.encoder_conv = nn.Sequential(

            nn.Conv3d(
                in_channels=num_output_neurons,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv3d(
                in_channels=out_channels,
                out_channels=2 * out_channels,
                kernel_size=kernel_size,
                padding=1
            ),
            nn.ReLU()
        )

        self.decoder_conv = nn.Sequential(

            nn.Conv3d(
                in_channels=2 * out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv3d(
                in_channels=out_channels,
                out_channels=num_output_neurons,
                kernel_size=kernel_size,
                padding=1
            )
        )


        self.encoder_linear = nn.Linear(
            2 * out_channels * points_per_dim ** 3,
            latent_dim
        )

        self.decoder_linear = nn.Linear(
            latent_dim,
            2 * out_channels * points_per_dim ** 3
        )



    def encode(self, x):

        x = self.encoder_conv(x)
        #before: (batch, num_output_neurons, points_per_dim, points_per_dim, points_per_dim)
        #after: (batch, 2*out_channels, points_per_dim, points_per_dim, points_per_dim)

        x = x.flatten(start_dim=1)
        #before: (batch, 2*out_channels, points_per_dim, points_per_dim, points_per_dim)
        #after: (batch, 2*out_channels * points_per_dim**3)

        z = self.encoder_linear(x)
        #before: (batch, 2*out_channels * points_per_dim**3)
        #after: (latent_dim)

        return z

    def decode(self, z):

        x = self.decoder_linear(z)
        #before: (latent_dim)
        #after: (batch, 2*out_channels * points_per_dim**3)

        x = x.view(
            -1,
            2 * self.out_channels,
            self.points_per_dim,
            self.points_per_dim,
            self.points_per_dim
        )
        #before: (batch, 2*out_channels * points_per_dim**3)
        #after: (batch, 2*out_channels, points_per_dim, points_per_dim, points_per_dim)

        x = self.decoder_conv(x)
        #before: (batch, 2*out_channels, points_per_dim, points_per_dim, points_per_dim)
        #after: (batch, num_output_neurons, points_per_dim, points_per_dim, points_per_dim)

        return x

    def forward(self, x):

        z = self.encode(x)

        x_hat = self.decode(z)

        return x_hat