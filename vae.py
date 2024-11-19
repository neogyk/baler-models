from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class bVAE(nn.Module):
    """
    beta - Variational Autoencoder (VAE) class.

    Args:
        input_dim (int): Dimensionality of the input data.
        hidden_dim (int): Dimensionality of the hidden layer.
        latent_dim (int): Dimensionality of the latent space.
    """

    def __init__(self, in_dim: int, latent_dim: int, out_dim: int):
        super(bVAE, self).__init__()
        print("Initialize the VAE")

        self.hidden_dim = [latent_dim, latent_dim, latent_dim]
        self.latent_dim = 2 * latent_dim
        self.output_dim = out_dim

        self.encoder = torch.nn.ModuleList(
            [
                nn.Linear(in_dim, h_dim)
                for in_dim, h_dim in zip(
                    [in_dim] + self.hidden_dim[:],
                    self.hidden_dim[:] + [self.latent_dim],
                )
            ]
        )
        self.encoder_norm = torch.nn.ModuleList(
            [nn.LayerNorm(h_dim) for h_dim in self.hidden_dim[:] + [self.latent_dim]]
        )

        self.softplus = nn.Softplus()

        self.decoder = torch.nn.ModuleList(
            [
                nn.Linear(in_dim, h_dim)
                for in_dim, h_dim in zip(
                    [self.latent_dim // 2] + self.hidden_dim[:],
                    self.hidden_dim[:] + [out_dim],
                )
            ]
        )
        self.decoder_norm = torch.nn.ModuleList(
            [nn.LayerNorm(h_dim) for h_dim in self.hidden_dim[:] + [out_dim]]
        )

    def encode(self, x):
        """
        Encodes the input data into the latent space.

        Args:
            x (torch.Tensor): Input data.
            eps (float): Small value to avoid numerical instability.

        Returns:
            torch.distributions.MultivariateNormal: Normal distribution of the encoded data.
        """
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            x = self.encoder_norm[i](x)
            x = torch.nn.functional.selu(x)

            # activation function
        mu, logvar = torch.chunk(x, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar, eps: float = 1e-8):
        """
        Reparameterizes the encoded data to sample from the latent space.

        Args:
            dist (torch.distributions.MultivariateNormal): Normal distribution of the encoded data.
        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)
        return torch.distributions.MultivariateNormal(
            mu, scale_tril=scale_tril
        ).rsample()

    def decode(self, z):
        """
        Decodes the data from the latent space to the original input space.

        Args:
            z (torch.Tensor): Data in the latent space.

        Returns:
            torch.Tensor: Reconstructed data in the original input space.
        """
        for i in range(len(self.decoder)):
            z = self.decoder[i](z)
            z = self.decoder_norm[i](z)
            if i < len(self.decoder) - 1:
                z = torch.nn.functional.selu(z)
        return z

    def forward(self, x, compute_loss: bool = True):
        """
        Performs a forward pass of the VAE.

        Args:
            x (torch.Tensor): Input data.
            compute_loss (bool): Whether to compute the loss or not.

        Returns:
            VAEOutput: VAE output dataclass.
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x = self.decode(z)
        return x  # , mu, log_var]

