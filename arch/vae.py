'''Simple classifier model. Credit goes to Samuel Lavoie
'''

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .modules.densenet import FullyConnectedNet
from .classifier import classify
from .utils import cross_correlation


logger = logging.getLogger('cortex.arch' + __name__)

mnist_encoder_args_ = dict(dim_h=64, batch_norm=True, f_size=5, pad=2, stride=2, min_dim=7, dim_out=1028)
mnist_decoder_args_ = dict(dim_h=64, batch_norm=True, f_size=4, pad=1, stride=2, n_steps=2)
convnet_encoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3, dim_out=1028)
convnet_decoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)


def setup(model=None, data=None, **kwargs):
    data['noise_variables']['z'] = (data['noise_variables']['z'][0], model['dim_z'])


class VAE(nn.Module):
    def __init__(self, encoder, decoder, dim_out=None, dim_z=None):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.mu_net = nn.Linear(dim_out, dim_z)
        self.logvar_net = nn.Linear(dim_out, dim_z)
        self.decoder = decoder
        self.mu = None
        self.logvar = None
        self.latent = None

    def reparametrize(self, mu, std):
        if self.training:
            esp = Variable(std.data.new(std.size()).normal_(), requires_grad=False).cuda()
            return mu + std * esp
        else:
            return mu

    def forward(self, x, nonlinearity=None):
        encoded = self.encoder(x, nonlinearity=F.relu)
        self.mu = self.mu_net(encoded)
        self.std = self.logvar_net(encoded).exp_()
        self.latent = self.reparametrize(self.mu, self.std)
        return self.decoder(self.latent, nonlinearity=nonlinearity)


def vae_routine(data, models, losses, results, viz, criterion=None, beta_kld=1.):
    X, Y, Z = data.get_batch('images', 'targets', 'z')
    vae_net = models['vae']

    outputs = vae_net(X, nonlinearity=F.tanh)
    gen = vae_net.decoder(Z, nonlinearity=F.tanh)

    r_loss = criterion(outputs, X, size_average=False)
    kl = 0.5 * (vae_net.std ** 2 + vae_net.mu ** 2 - 2. * torch.log(vae_net.std) - 1.).sum()
    losses.update(vae=r_loss + beta_kld * kl)
    correlations = cross_correlation(vae_net.mu, remove_diagonal=True)

    results.update(KL_divergence=kl.data[0])
    viz.add_image(outputs, name='reconstruction')
    viz.add_image(gen, name='generated')
    viz.add_image(X, name='ground truth')
    viz.add_heatmap(correlations.data, name='latent correlations')
    viz.add_scatter(vae_net.mu.data, labels=Y.data, name='latent values')


def classifier_routine(data, models, losses, results, viz, **kwargs):
    X, Y = data.get_batch('images', 'targets')
    vae_net = models['vae']
    classifier = models['classifier']

    vae_net(X, nonlinearity=F.tanh)
    classify(classifier, vae_net.mu, Y, losses=losses, results=results, **kwargs)


def build_model(data, models, model_type='convnet', dim_z=64, encoder_args=None, decoder_args=None):
    encoder_args = encoder_args or {}
    decoder_args = decoder_args or {}
    shape = data.get_dims('x', 'y', 'c')
    dim_l = data.get_dims('labels')

    if model_type == 'convnet':
        from .modules.convnets import SimpleConvEncoder as Encoder
        from .modules.conv_decoders import SimpleConvDecoder as Decoder
        encoder_args_ = convnet_encoder_args_
        decoder_args_ = convnet_decoder_args_
    elif model_type == 'mnist':
        from .modules.convnets import SimpleConvEncoder as Encoder
        from .modules.conv_decoders import SimpleConvDecoder as Decoder
        encoder_args_ = mnist_encoder_args_
        decoder_args_ = mnist_decoder_args_
    else:
        raise NotImplementedError(model_type)

    encoder_args_.update(**encoder_args)
    decoder_args_.update(**decoder_args)
    if shape[0] == 64:
        encoder_args_['n_steps'] = 4
        decoder_args_['n_steps'] = 4

    encoder = Encoder(shape, **encoder_args_)
    decoder = Decoder(shape, dim_in=dim_z, **decoder_args_)
    vae = VAE(encoder, decoder, dim_out=encoder_args_['dim_out'], dim_z=dim_z)
    classifier = FullyConnectedNet(dim_z, dim_h=[64, 64], dim_out=dim_l, batch_norm=True, dropout=0.2)

    models.update(vae=vae, classifier=classifier)


ROUTINES = dict(vae=vae_routine, classifier=classifier_routine)

DEFAULT_CONFIG = dict(
    data=dict(batch_size=dict(train=64, test=640),
              noise_variables=dict(z=('normal', 64))),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=1e-4,
    ),
    model=dict(dim_z=64, encoder_args=None, decoder_args=None),
    routines=dict(vae=dict(criterion=F.mse_loss, beta_kld=1.),
                  classifier=dict(criterion=nn.CrossEntropyLoss())),
    train=dict(
        epochs=500,
        archive_every=10
    )
)