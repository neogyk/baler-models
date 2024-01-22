import torch
from torch import nn

class dense_demo(nn.Module):
    def __init__(self, input_size, latent_size, *args, **kwargs):
        super(dense_demo, self).__init__(*args, **kwargs)

        self.encoder = torch.nn.Sequential(
            nn.Linear(input_size, 200, dtype=torch.float),
            nn.LeakyReLU(),
            nn.Linear(200, 100, dtype=torch.float),
            nn.LeakyReLU(),
            nn.Linear(100, 50, dtype=torch.float),
            nn.LeakyReLU(),
            nn.Linear(50, latent_size, dtype=torch.float),
        )

        self.decoder = torch.nn.Sequential(
            nn.Linear(latent_size, 50, dtype=torch.float),
            nn.LeakyReLU(),
            nn.Linear(50, 100, dtype=torch.float),
            nn.LeakyReLU(),
            nn.Linear(100, 200, dtype=torch.float),
            nn.LeakyReLU(),
            nn.Linear(200, input_size, dtype=torch.float),
        )

    def encode(self, input_data):
        latent_space = self.encoder(input_data)
        return latent_space

    def decode(self, latent_space):
        reconstruction = self.decoder(latent_space)
        return reconstruction

    def forward(self, input_data):
        latent_space = self.encoder(input_data)
        reconstructed = self.decoder(latent_space)
        return reconstructed
