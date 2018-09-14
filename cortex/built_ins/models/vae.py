'''Simple Variational Autoencoder model.
'''

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from cortex.built_ins.models.utils import ms_ssim
from cortex.built_ins.models.image_coders import ImageDecoder, ImageEncoder
from cortex.plugins import ModelPlugin, register_plugin


__author__ = 'R Devon Hjelm and Samuel Lavoie'
__author_email__ = 'erroneus@gmail.com'

logger = logging.getLogger('cortex.vae')


class VAENetwork(nn.Module):
    '''VAE model.

    Attributes:
        encoder: Encoder network.
        mu_net: Single layer network for caculating mean.
        logvar_net: Single layer network for calculating log variance.
        decoder: Decoder network.
        mu: The mean after encoding.
        logvar: The log variance after encoding.
        latent: The latent state (Z).

    '''

    def __init__(self, encoder, decoder, dim_out=None, dim_z=None):
        super(VAENetwork, self).__init__()
        self.encoder = encoder
        self.mu_net = nn.Linear(dim_out, dim_z)
        self.logvar_net = nn.Linear(dim_out, dim_z)
        self.decoder = decoder
        self.mu = None
        self.logvar = None
        self.latent = None

    def encode(self, inputs, **kwargs):
        encoded = self.encoder(inputs, **kwargs)
        encoded = F.relu(encoded)
        return self.mu_net(encoded)

    def reparametrize(self, mu, std):
        if self.training:
            esp = Variable(
                std.data.new(std.size()).normal_(), requires_grad=False).cuda()
            return mu + std * esp
        else:
            return mu

    def forward(self, x, nonlinearity=None):
        encoded = self.encoder(x)
        encoded = F.relu(encoded)
        self.mu = self.mu_net(encoded)
        self.std = self.logvar_net(encoded).exp_()
        self.latent = self.reparametrize(self.mu, self.std)
        return self.decoder(self.latent, nonlinearity=nonlinearity)


class VAE(ModelPlugin):
    '''Variational autoencder.

    A generative model trained using the variational lower-bound to the
    log-likelihood.
    See: Kingma, Diederik P., and Max Welling.
    "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).

    '''

    defaults = dict(
        data=dict(
            batch_size=dict(train=64, test=640), inputs=dict(inputs='images')),
        optimizer=dict(optimizer='Adam', learning_rate=1e-4),
        train=dict(save_on_lowest='losses.vae'))

    def __init__(self):
        super().__init__()

        self.encoder = ImageEncoder(contract=dict(
            kwargs=dict(dim_out='dim_encoder_out')))
        decoder_contract = dict(kwargs=dict(dim_in='dim_z'))
        self.decoder = ImageDecoder(contract=decoder_contract)

    def build(self, dim_z=64, dim_encoder_out=1024):
        '''

        Args:
            dim_z: Latent dimension.
            dim_encoder_out: Dimension of the final layer of the decoder before
                             decoding to mu and log sigma.

        '''
        self.encoder.build()
        self.decoder.build()

        self.add_noise('Z', dist='normal', size=dim_z)
        encoder = self.nets.encoder
        decoder = self.nets.decoder
        vae = VAENetwork(encoder, decoder, dim_out=dim_encoder_out, dim_z=dim_z)
        self.nets.vae = vae

    def routine(self, inputs, targets, Z, vae_criterion=F.mse_loss,
                beta_kld=1.):
        '''

        Args:
            vae_criterion: Reconstruction criterion.
            beta_kld: Beta scaling for KL term in lower-bound.

        '''

        vae = self.nets.vae
        outputs = vae(inputs)

        try:
            r_loss = vae_criterion(
                outputs, inputs, size_average=False) / inputs.size(0)
        except RuntimeError as e:
            logger.error('Runtime error. This could possibly be due to using '
                         'the wrong encoder / decoder for this dataset. '
                         'If you are using MNIST, for example, use the '
                         'arguments `--encoder_type mnist --decoder_type '
                         'mnist`')
            raise e

        kl = (0.5 * (vae.std**2 + vae.mu**2 - 2. * torch.log(vae.std) -
                     1.).sum(1).mean())

        msssim = ms_ssim(inputs, outputs)

        self.losses.vae = (r_loss + beta_kld * kl)
        self.results.update(KL_divergence=kl.item(), ms_ssim=msssim.item())

    def visualize(self, inputs, targets, Z):
        vae = self.nets.vae

        outputs = vae(inputs)

        self.add_image(outputs, name='reconstruction')
        self.add_image(inputs, name='ground truth')
        self.add_scatter(vae.mu.data, labels=targets.data, name='latent values')
        self.decoder.visualize(Z)


register_plugin(VAE)
