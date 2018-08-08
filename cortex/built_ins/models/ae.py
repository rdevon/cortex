'''Module for autoencoder model.

'''

import torch.nn.functional as F

from cortex.plugins import ModelPlugin, register_plugin
from cortex.built_ins.models.image_coders import ImageEncoder, ImageDecoder
from cortex.built_ins.networks.ae_network import AENetwork


class Autoencoder(ModelPlugin):
    '''Simple autoencder model.

        Trains a noiseless autoencoder of image data.

    '''
    defaults = dict(
        data=dict(
            batch_size=dict(train=64, test=64), inputs=dict(inputs='images')),
        optimizer=dict(optimizer='Adam', learning_rate=1e-4),
        train=dict(save_on_lowest='losses.ae'))

    def __init__(self, Encoder=None, Decoder=None):
        super().__init__()
        if Encoder is None:
            Encoder = ImageEncoder
        if Decoder is None:
            Decoder = ImageDecoder
        self.encoder = Encoder()
        self.decoder = Decoder()

    def build(self, dim_z=64):
        self.encoder.build(dim_out=dim_z)
        self.decoder.build(dim_in=dim_z)

        encoder = self.nets.encoder
        decoder = self.nets.decoder
        ae = AENetwork(encoder, decoder)
        self.nets.ae = ae

    def routine(self, inputs, targets, ae_criterion=F.mse_loss):
        '''

        Args:
            ae_criterion: Criterion for the autoencoder.

        '''
        ae = self.nets.ae
        outputs = ae(inputs)
        r_loss = ae_criterion(
            outputs, inputs, size_average=False) / inputs.size(0)
        self.losses.ae = r_loss

    def visualize(self, inputs, targets):
        ae = self.nets.ae
        outputs = ae(inputs)
        self.add_image(outputs, name='reconstruction')
        self.add_image(inputs, name='ground truth')


register_plugin(Autoencoder)
