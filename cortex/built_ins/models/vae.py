'''Simple Variational Autoencoder model.
'''

from cortex.plugins import (register_plugin, BuildPlugin, ModelPlugin,
                            RoutinePlugin)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from cortex.built_ins.models.classifier import (SimpleClassifierBuild,
                                                ClassifyRoutine)

from .utils import update_encoder_args, update_decoder_args


__author__ = 'R Devon Hjelm and Samuel Lavoie'
__author_email__ = 'erroneus@gmail.com'


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

    def reparametrize(self, mu, std):
        if self.training:
            esp = Variable(std.data.new(std.size()).normal_(),
                           requires_grad=False).cuda()
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


class VAERoutine(RoutinePlugin):
    '''VAE for training the VAE.

    '''
    plugin_name = 'VAE'
    plugin_nets = ['vae']
    plugin_vars = ['input', 'noise', 'targets', 'encoder_mean']

    def run(self, vae_criterion=F.mse_loss, beta_kld=1.):
        '''

        Args:
            vae_criterion: Reconstruction criterion.
            beta_kld: Beta scaling for KL term in lower-bound.

        '''
        X = self.vars.input
        Y = self.vars.targets
        Z = self.vars.noise

        vae_net = self.nets.vae

        outputs = vae_net(X)
        outputs = F.tanh(outputs)
        gen = vae_net.decoder(Z)
        gen = F.tanh(gen)

        r_loss = vae_criterion(outputs, X, size_average=False) / X.size(0)
        kl = (0.5 * (vae_net.std ** 2 + vae_net.mu ** 2 - 2. *
                     torch.log(vae_net.std) - 1.).sum(1).mean())

        self.losses.vae = (r_loss + beta_kld * kl)
        self.results.update(KL_divergence=kl.item())

        self.add_image(outputs, name='reconstruction')
        self.add_image(gen, name='generated')
        self.add_image(X, name='ground truth')
        self.add_scatter(vae_net.mu.data, labels=Y.data, name='latent values')
        self.vars.encoder_mean = vae_net.mu.detach()


class ImageEncoderBuild(BuildPlugin):
    '''Builds a simple image encoder.

    '''
    plugin_name = 'image_encoder'
    plugin_nets = ['image_encoder']
    def build(self, encoder_type: str='convnet', dim_out: int=None,
              encoder_args={}):
        '''

        Args:
            encoder_type: Encoder model type.
            dim_out: Output size.
            encoder_args: Arguments for encoder build.

        '''
        x_shape = self.get_dims('x', 'y', 'c')
        Encoder, encoder_args = update_encoder_args(x_shape,
                                                    model_type=encoder_type,
                                                    encoder_args=encoder_args)
        encoder = Encoder(x_shape, dim_out=dim_out, **encoder_args)
        self.nets.image_encoder = encoder


class ImageDecoderBuild(BuildPlugin):
    '''Builds a simple image encoder.

    '''
    plugin_name = 'image_decoder'
    plugin_nets = ['image_decoder']

    def build(self, decoder_type: str='convnet', dim_in: int=64,
              decoder_args={}):
        '''

        Args:
            decoder_type: Decoder model type.
            dim_in: Input size.
            decoder_args: Arguments for the decoder.

        '''
        x_shape = self.get_dims('x', 'y', 'c')
        Decoder, decoder_args = update_decoder_args(x_shape,
                                                    model_type=decoder_type,
                                                    decoder_args=decoder_args)
        decoder = Decoder(x_shape, dim_in=dim_in, **decoder_args)
        self.nets.image_decoder = decoder


class VAEBuild(BuildPlugin):
    '''Builds a VAE

    This model builds on encoders and decoders build before it.

    '''
    plugin_name = 'VAE'
    plugin_nets = ['encoder', 'decoder', 'vae']

    def build(self, dim_z=64, dim_encoder_out=1028):
        '''

        Args:
            dim_z: Latent dimension.
            dim_encoder_out: Dimension of the final layer of the decoder before
                             decoding to mu and log sigma.

        '''
        self.add_noise('z', dist='normal', size=dim_z)
        encoder = self.nets.encoder
        decoder = self.nets.decoder
        vae = VAENetwork(encoder, decoder, dim_out=dim_encoder_out,
                         dim_z=dim_z)
        self.nets.vae = vae


class VAE(ModelPlugin):
    '''Variational autoencder.

    A generative model trained using the variational lower-bound to the
    log-likelihood.
    See: Kingma, Diederik P., and Max Welling.
    "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).

    '''
    plugin_name = 'VAE'

    data_defaults = dict(batch_size=dict(train=64, test=640))
    optimizer_defaults = dict(optimizer='Adam', learning_rate=1e-4)
    train_defaults = dict(save_on_lowest='losses.vae')

    def __init__(self, add_classification=True):
        '''

        Args:
            add_classification: Adds a classifier on top of the latents.

        '''
        super().__init__()
        self.builds.encoder = ImageEncoderBuild(dim_out='dim_encoder_out',
                                                image_encoder='encoder')
        self.builds.decoder = ImageDecoderBuild(dim_in='dim_z',
                                                image_decoder='decoder')
        self.builds.vae = VAEBuild()
        self.routines.vae = VAERoutine(input='data.images', noise='data.z',
                                       targets='data.targets')

        if add_classification:
            self.builds.classifier = SimpleClassifierBuild(dim_in='dim_z')
            self.routines.classify = ClassifyRoutine(
                classifier='simple_classifier', inputs='encoder_mean',
                targets='data.targets', images='data.images')
            self.add_train_procedure(self.routines.vae, self.routines.classify)
        else:
            self.add_train_procedure(self.routines.vae)


register_plugin(VAE)
