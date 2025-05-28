import torch
from gbn_layer import GBN
from typing import Literal
import pdb
from torch.distributions import Normal, Bernoulli
import utils
import rans
import numpy as np

rng = np.random.RandomState(0)

_LAYER_TYPES = Literal["dense", "convolutional"]
_ARCHITECTURE_TYPES = Literal["forward", "unet"]


def beta_binomial_log_pdf(k, n, alpha, beta):
    numer = (
        lgamma(n + 1) + lgamma(k + alpha) + lgamma(n - k + beta) + lgamma(alpha + beta)
    )
    denom = (
        lgamma(k + 1)
        + lgamma(n - k + 1)
        + lgamma(n + alpha + beta)
        + lgamma(alpha)
        + lgamma(beta)
    )
    return numer - denom


class betaVAE(torch.nn.Module):
    """
    beta - Variational Autoencoder (VAE) class.

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
    """

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        layer_type: _LAYER_TYPES = "convolutional",
        architecture_type: _ARCHITECTURE_TYPES = "forward",
        bitback=True,
    ):
        super(betaVAE, self).__init__()
        self.layer_type: _LAYER_TYPES = layer_type
        self.architecture_type: _ARCHITECTURE_TYPES = architecture_type
        self.bitback = bitback
        self.hidden_dim = [1, 32, 1]
        self.hidden_dim_decoder = [1, 32, 1]
        self.latent_dim = latent_dim
        self.output_dim = in_dim
        self.mu_l = torch.nn.Linear(latent_dim, latent_dim)
        self.var_l = torch.nn.Linear(latent_dim, latent_dim)

        self.encoder_modules = torch.nn.ModuleList(
            [
                (
                    torch.nn.Linear(in_dim, h_dim)
                    if self.layer_type is "dense"
                    else torch.nn.Conv2d(
                        in_dim, out_channels=h_dim, kernel_size=3, stride=5, padding=0
                    )
                )
                for in_dim, h_dim in zip(
                    [in_dim] + self.hidden_dim[:], self.hidden_dim[:]
                )
            ]
        )
        self.encoder_norm = torch.nn.ModuleList(
            [
                (
                    GBN(h_dim)
                    if self.layer_type is "dense"
                    else torch.nn.BatchNorm2d(h_dim)
                )
                for h_dim in self.hidden_dim[:]
            ]
        )

        self.decoder_modules = torch.nn.ModuleList(
            [
                (
                    torch.nn.Linear(in_dim, h_dim)
                    if self.layer_type is "dense"
                    else torch.nn.ConvTranspose2d(
                        in_dim,
                        h_dim,
                        kernel_size=3,
                        stride=5,
                        padding=0,
                        output_padding=0,
                    )
                )
                for in_dim, h_dim in zip(
                    self.hidden_dim_decoder[:],
                    self.hidden_dim_decoder[1:] + [self.output_dim],
                )
            ]
        )
        self.decoder_norm = torch.nn.ModuleList(
            [
                (
                    GBN(h_dim)
                    if self.layer_type is "dense"
                    else torch.nn.BatchNorm2d(h_dim)
                )
                for h_dim in self.hidden_dim_decoder[1:] + [self.output_dim]
            ]
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
        for i in range(len(self.encoder_modules)):
            x = self.encoder_modules[i](x)
            print("X: ", x.size())
            x = self.encoder_norm[i](x)
            x = torch.nn.functional.leaky_relu(x)

        if self.layer_type == "convolutional":
            x = torch.flatten(x, start_dim=1)

        mu, logvar = torch.nn.functional.leaky_relu(
            self.mu_l(x)
        ), torch.nn.functional.leaky_relu(self.var_l(x))
        return mu, logvar

    def reparameterize(self, mu, logvar, eps: float = 1e-6):
        """
        Reparameterizes the encoded data to sample from the latent space.

        Args:
            dist (torch.distributions.MultivariateNormal): Normal distribution of the encoded data.
        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = eps * std + mu
        return latent

    def decode(self, z: torch.Tensor):
        """
        Decodes the data from the latent space to the original input space.

        Args:
            z (torch.Tensor): Data in the latent space.

        Returns:
            torch.Tensor: Reconstructed data in the original input space.
        """
        for i in range(len(self.decoder_modules)):
            z = self.decoder_modules[i](z)
            z = self.decoder_norm[i](z)
            z = torch.nn.functional.leaky_relu(z)

        if self.bitback:
            z = torch.nn.functional.tanh(z)
        else:
            z = torch.nn.functional.sigmoid(z)
        return z

    def loss(self, x):
        mu, log_var = self.encode(x)
        x = self.reparameterize(mu, logvar=log_var)
        x_probs = self.encode(x)
        dist = Bernoulli(x_probs)
        l = torch.sum(dist.log_prob(x.view(-1, 1)), dim=1)
        p_z = torch.sum(Normal(0, 1).log_prob(z), dim=1)
        q_z = torch.sum(Normal(z_mu, z_std).log_prob(z), dim=1)
        return -torch.mean(l + p_z - q_z) * np.log2(np.e)

    def forward(self, x):
        """
        Performs a forward pass of the VAE.

        Args:
            x (torch.Tensor): Input data.
        Returns:
            VAEOutput: VAE output dataclass.
        """
        mu, logvar = self.encode(x)
        self.mu, self.logvar = mu, logvar
        z = self.reparameterize(mu, logvar)
        x = self.decode(z)
        return x

    def compress(self, x):
        x = [_x.view(-1) for _x in x]
        other_bits = rng.randint(low=1 << 16, high=1 << 31, size=20, dtype=np.uint32)
        state = rans.unflatten(other_bits)

        for _x in x:
            state = utils.vae_append(state, _x)

        compressed_message = rans.flatten(state)
        return compressed_message

    def decompress(self, z):
        state = rans.unflatten(z)
        state, x = utils.vae_pop(state)
        return x


if __name__ == "__main__":

    vae = betaVAE(256, latent_dim=4, layer_type="dense")

    # Test the compression:
    x = torch.rand(32, 3, 256, 256)
    compressed = vae.compress(x)
