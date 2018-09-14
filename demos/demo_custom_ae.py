import torch.nn.functional as F
from torch import nn

from cortex.plugins import ModelPlugin
from cortex.main import run


class Autoencoder(nn.Module):
    """
    Encapsulation of an encoder and a decoder.
    """
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, nonlinearity=None):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AE(ModelPlugin):
    """
    Autoencoder designed with Pytorch components.
    """
    defaults = dict(
        data=dict(
            batch_size=dict(train=64, test=64), inputs=dict(inputs='images')),
        optimizer=dict(optimizer='Adam', learning_rate=1e-4),
        train=dict(save_on_lowest='losses.ae'))

    def __init__(self):
        super().__init__()

    def build(self, dim_z=64, dim_encoder_out=64):
        """
        Build AE with an encoder and decoder and attribute
        an Autoencoder to self.nets.
        Args:
            dim_z: int
            dim_encoder_out: in

        Returns: None

        """
        encoder = nn.Sequential(
            nn.Linear(28, 256),
            nn.ReLU(True),
            nn.Linear(256, 28),
            nn.ReLU(True))
        decoder = nn.Sequential(
            nn.Linear(28, 256),
            nn.ReLU(True),
            nn.Linear(256, 28),
            nn.Sigmoid())
        self.nets.ae = Autoencoder(encoder, decoder)

    def routine(self, inputs, ae_criterion=F.mse_loss):
        """
        Training routine and loss computing.
        Args:
            inputs: torch.Tensor
            ae_criterion: function

        Returns: None

        """
        encoded = self.nets.ae.encoder(inputs)
        outputs = self.nets.ae.decoder(encoded)
        r_loss = ae_criterion(
            outputs, inputs, size_average=False) / inputs.size(0)
        self.losses.ae = r_loss

    def visualize(self, inputs):
        """
        Adding generated images and base images to
        visualization.
        Args:
            inputs: torch.Tensor
        Returns: None

        """
        encoded = self.nets.ae.encoder(inputs)
        outputs = self.nets.ae.decoder(encoded)
        self.add_image(outputs, name='reconstruction')
        self.add_image(inputs, name='ground truth')


if __name__ == '__main__':
    autoencoder = AE()
    run(model=autoencoder)
