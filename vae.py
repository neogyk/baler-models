import torch


class bVAE(torch.nn.Module):
    """
    beta - Variational Autoencoder (VAE) class.

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
    """

    def __init__(self, in_dim: int, latent_dim: int):
        super(bVAE, self).__init__()
        self.hidden_dim = [512, 512, 256, 256]
        self.hidden_dim_decoder = [latent_dim, 256, 512, 512]
        self.latent_dim = latent_dim
        self.output_dim = in_dim
        self.mu_l = torch.nn.Linear(self.hidden_dim[-1], latent_dim)
        self.var_l = torch.nn.Linear(self.hidden_dim[-1], latent_dim)

        self.encoder_modules = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_dim, h_dim)
                for in_dim, h_dim in zip(
                    [in_dim] + self.hidden_dim[:], self.hidden_dim[:]
                )
            ]
        )
        self.encoder_norm = torch.nn.ModuleList(
            [
                torch.nn.InstanceNorm1d(h_dim) for h_dim in self.hidden_dim[:]
            ]  # + [self.latent_dim]]
        )

        self.softplus = torch.nn.Softplus()

        self.decoder_modules = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_dim, h_dim)
                for in_dim, h_dim in zip(
                    [self.latent_dim] + self.hidden_dim_decoder[:],
                    self.hidden_dim_decoder[:] + [self.output_dim],
                )
            ]
        )
        self.decoder_norm = torch.nn.ModuleList(
            [
                torch.nn.InstanceNorm1d(h_dim)
                for h_dim in self.hidden_dim_decoder[:] + [self.output_dim]
            ]
        )

    def encoder(self, x):
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

            x = self.encoder_norm[i](x)
            x = torch.nn.functional.leaky_relu(x)

        mu, logvar = torch.nn.functional.leaky_relu(
            self.mu_l(x)
        ), torch.nn.functional.leaky_relu(self.var_l(x))
        return mu, logvar

    def reparameterize(self, mu, logvar, eps: float = 1e-4):
        """
        Reparameterizes the encoded data to sample from the latent space.

        Args:
            dist (torch.distributions.MultivariateNormal): Normal distribution of the encoded data.
        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + eps * std

    def decoder(self, z):
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
        # z = torch.nn.functional.relu(z)
        return z

    def forward(self, x):
        """
        Performs a forward pass of the VAE.

        Args:
            x (torch.Tensor): Input data.
        Returns:
            VAEOutput: VAE output dataclass.
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x = self.decoder(z)
        return x
