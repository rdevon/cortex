from cortex.main import run
from cortex.plugins import ModelPlugin
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from cortex.built_ins.models.utils import update_encoder_args, update_decoder_args, ms_ssim


class ImageEncoder(ModelPlugin):

    def build(self,
              dim_out,
              encoder_type: str = 'convnet',
              encoder_args=dict(fully_connected_layers=1028)):
        x_shape = self.get_dims('x', 'y', 'c')
        Encoder, encoder_args = update_encoder_args(
            x_shape, model_type=encoder_type, encoder_args=encoder_args)
        encoder = Encoder(x_shape, dim_out=dim_out, **encoder_args)
        self.nets.encoder = encoder

    def encode(self, inputs, **kwargs):
        return self.nets.encoder(inputs, **kwargs)

    def visualize(self, inputs, targets):
        Z = self.encode(inputs)
        if targets is not None:
            targets = targets.data
        self.add_scatter(Z.data, labels=targets, name='latent values')


class ImageDecoder(ModelPlugin):

    def build(self,
              dim_in,
              decoder_type: str = 'convnet',
              decoder_args=dict(output_nonlinearity='tanh')):
        x_shape = self.get_dims('x', 'y', 'c')
        Decoder, decoder_args = update_decoder_args(
            x_shape, model_type=decoder_type, decoder_args=decoder_args)
        decoder = Decoder(x_shape, dim_in=dim_in, **decoder_args)
        self.nets.decoder = decoder

    def routine(self, inputs, Z, decoder_crit=F.mse_loss):
        X = self.decode(Z)
        self.losses.decoder = decoder_crit(X, inputs) / inputs.size(0)
        msssim = ms_ssim(inputs, X)
        self.results.ms_ssim = msssim.item()

    def decode(self, Z):
        return self.nets.decoder(Z)

    def visualize(self, Z):
        gen = self.decode(Z)
        self.add_image(gen, name='generated')


class AENetwork(nn.Module):

    def __init__(self, encoder, decoder):
        super(AENetwork, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, nonlinearity=None):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AE(ModelPlugin):
    plugin_name = 'AE'
    defaults = dict(
        data=dict(
            batch_size=dict(train=64, test=64), inputs=dict(inputs='images')),
        optimizer=dict(optimizer='Adam', learning_rate=1e-4),
        train=dict(save_on_lowest='losses.ae'))

    def __init__(self):
        super().__init__()
        self.encoder = ImageEncoder()
        self.decoder = ImageDecoder()

    def build(self, dim_z=64, dim_encoder_out=64):
        self.encoder.build(dim_encoder_out)
        self.decoder.build(dim_z)
        encoder = self.nets.encoder
        decoder = self.nets.decoder
        ae = AENetwork(encoder, decoder)
        self.nets.ae = ae

    def routine(self, inputs, targets, ae_criterion=F.mse_loss):
        ae = self.nets.ae
        outputs = ae(inputs)
        r_loss = ae_criterion(outputs, inputs, size_average=False) / inputs.size(0)
        self.losses.ae = r_loss

    def visualize(self, inputs, targets):
        ae = self.nets.ae
        outputs = ae(inputs)
        self.add_image(outputs, name='reconstruction')
        self.add_image(inputs, name='ground truth')


if __name__ == '__main__':
    autoencoder = AE()
    run(model=autoencoder)
