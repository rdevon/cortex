import torch.nn as nn


class AENetwork(nn.Module):
    def __init__(self, encoder, decoder):
        super(AENetwork, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, nonlinearity=None):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
