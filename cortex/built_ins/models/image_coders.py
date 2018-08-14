from cortex.plugins import ModelPlugin
from cortex.built_ins.models.utils import update_encoder_args, update_decoder_args, ms_ssim
import torch.nn.functional as F


class ImageEncoder(ModelPlugin):

    def build(self,
              dim_out=None,
              encoder_type: str = 'convnet',
              encoder_args=dict(fully_connected_layers=1028),
              Encoder=None):
        x_shape = self.get_dims('x', 'y', 'c')
        Encoder_, encoder_args = update_encoder_args(
            x_shape, model_type=encoder_type, encoder_args=encoder_args)
        Encoder = Encoder or Encoder_
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
              dim_in=None,
              decoder_type: str = 'convnet',
              decoder_args=dict(output_nonlinearity='tanh'),
              Decoder=None):
        x_shape = self.get_dims('x', 'y', 'c')
        Decoder_, decoder_args = update_decoder_args(
            x_shape, model_type=decoder_type, decoder_args=decoder_args)
        Decoder = Decoder or Decoder_
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
