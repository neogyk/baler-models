import torch
from torch import nn

class TransformerAE(nn.Module):
    """Autoencoder mixed with the Transformer Encoder layer

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        in_dim,
        out_dim, 
        encoder_h_dim:list = [1024, 256, 128, 64],
        decoder_h_dim:list = [64, 128, 256, 1024],       
        nheads=1,
        latent_dim=64,
        activation=torch.nn.functional.gelu,
    ):
        super(TransformerAE, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim

        self.encoder_transformer_layers = torch.nn.ModuleList(
                [nn.TransformerEncoderLayer(
                    batch_first=True,
                    norm_first=True,
                    d_model = i,
                    activation = activation,
                    dim_feedforward = i,
                    nhead = nheads,
                    )
                 for i in ([in_dim] + encoder_h_dim[:]) 

                    ])

        self.encoder_linear_layers = torch.nn.ModuleList([
        torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(),
            torch.nn.Linear(i[0], i[-1]),
            torch.nn.GELU(),
        ) for i in zip([in_dim] + encoder_h_dim, encoder_h_dim + [latent_dim])
        ])
        
        self.decoder_transformer_layers = torch.nn.ModuleList(
                [nn.TransformerEncoderLayer(
                    batch_first=True,
                    norm_first=True,
                    d_model = i,
                    activation = activation,
                    dim_feedforward = i,
                    nhead = nheads,
                    
                    )
                 for i in ([latent_dim] + decoder_h_dim[:]) 

                    ])

        self.decoder_linear_layers = torch.nn.ModuleList([
        torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(),
            torch.nn.Linear(i[0], i[-1]),
            torch.nn.GELU(),
        ) for i in zip([latent_dim] + decoder_h_dim, decoder_h_dim+ [out_dim])
        ])


    def encoder(self, x: torch.Tensor):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        for i, layer in enumerate(self.encoder_transformer_layers):
            x = self.encoder_transformer_layers[i](x)
            x = self.encoder_linear_layers[i](x)
        assert x.size()[-1] == self.latent_dim
        return x
        

    def decoder(self, x: torch.Tensor):
        """_summary_

        Args:
            z (_type_): _description_

        Returns:
            _type_: _description_
        """
        for i, layer in enumerate(self.decoder_transformer_layers):
            x = self.decoder_transformer_layers[i](x)
            x = self.decoder_linear_layers[i](x)
        assert x.size()[-1] == self.out_dim
        return x

    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            z (_type_): _description_

        Returns:
            _type_: _description_
        """
        z = self.encoder(x)
        x = self.decoder(z)
        return x
