'''Simple classifier model. Credit goes to Samuel Lavoie
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .modules.fully_connected import FullyConnectedNet
from .classifier import classify
from .utils import cross_correlation


resnet_encoder_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=3)
resnet_decoder_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=3)
mnist_encoder_args_ = dict(dim_h=64, batch_norm=True, f_size=5, pad=2, stride=2, min_dim=7)
mnist_decoder_args_ = dict(dim_h=64, batch_norm=True, f_size=4, pad=1, stride=2, n_steps=2)
convnet_encoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)
convnet_decoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)


def setup(model=None, data=None, **kwargs):
    data['noise_variables']['z']['size'] = model['dim_z']


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

    results.update(KL_divergence=kl.item())
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


def update_encoder_args(x_shape, model_type='convnet', encoder_args=None):
    encoder_args = encoder_args or {}

    if model_type == 'resnet':
        from .modules.resnets import ResEncoder as Encoder
        encoder_args_ = resnet_encoder_args_
    elif model_type == 'convnet':
        from .modules.convnets import SimpleConvEncoder as Encoder
        encoder_args_ = convnet_encoder_args_
    elif model_type == 'mnist':
        from .modules.convnets import SimpleConvEncoder as Encoder
        encoder_args_ = mnist_encoder_args_
    else:
        raise NotImplementedError(model_type)

    encoder_args_.update(**encoder_args)
    if x_shape[0] == 64:
        encoder_args_['n_steps'] = 4
    elif x_shape[0] == 128:
        encoder_args_['n_steps'] = 5

    return Encoder, encoder_args_


def update_decoder_args(x_shape, model_type='convnet', decoder_args=None):
    decoder_args = decoder_args or {}

    if model_type == 'resnet':
        from .modules.resnets import ResDecoder as Decoder
        decoder_args_ = resnet_decoder_args_
    elif model_type == 'convnet':
        from .modules.conv_decoders import SimpleConvDecoder as Decoder
        decoder_args_ = convnet_decoder_args_
    elif model_type == 'mnist':
        from .modules.conv_decoders import SimpleConvDecoder as Decoder
        decoder_args_ = mnist_decoder_args_
    else:
        raise NotImplementedError(model_type)

    decoder_args_.update(**decoder_args)
    if x_shape[0] == 64:
        decoder_args_['n_steps'] = 4
    elif x_shape[0] == 128:
        decoder_args_['n_steps'] = 5

    return Decoder, decoder_args_


def build_encoder(models, x_shape, dim_out, Encoder, key='encoder', **encoder_args):
    logger.debug('Forming encoder with class {} and args: {}'.format(Encoder, encoder_args))
    encoder = Encoder(x_shape, dim_out=dim_out, **encoder_args)

    if models is not None:
        models[key] = encoder

    return encoder


def build_decoder(models, x_shape, dim_in, Decoder, key='decoder', **decoder_args):
    logger.debug('Forming dencoder with class {} and args: {}'.format(Decoder, decoder_args))
    decoder = Decoder(x_shape, dim_in=dim_in, **decoder_args)

    if models is not None:
        models[key] = decoder

    return decoder


def build_model(data, models, model_type='convnet', dim_z=64, dim_encoder_out=1028, encoder_args=None,
                decoder_args=None):
    x_shape = data.get_dims('x', 'y', 'c')
    dim_l = data.get_dims('labels')

    Encoder, encoder_args = update_encoder_args(x_shape, model_type=model_type, encoder_args=encoder_args)
    Decoder, decoder_args = update_decoder_args(x_shape, model_type=model_type, decoder_args=decoder_args)

    encoder = build_encoder(None, x_shape, dim_encoder_out, Encoder, **encoder_args)
    decoder = build_decoder(None, x_shape, dim_z, Decoder, **decoder_args)
    vae = VAE(encoder, decoder, dim_out=dim_encoder_out, dim_z=dim_z)

    classifier = FullyConnectedNet(dim_z, dim_h=[200, 200], dim_out=dim_l, batch_norm=True, dropout=0.2)
    models.update(vae=vae, classifier=classifier)


ROUTINES = dict(vae=vae_routine, classifier=classifier_routine)

DEFAULT_CONFIG = dict(
    data=dict(batch_size=dict(train=64, test=640),
              noise_variables=dict(z=dict(dist='normal', size=64))),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=1e-4,
    ),
    model=dict(dim_z=64, model_type='convnet', dim_encoder_out=1028, encoder_args=None, decoder_args=None),
    routines=dict(vae=dict(criterion=F.mse_loss, beta_kld=1.),
                  classifier=dict(criterion=nn.CrossEntropyLoss())),
    train=dict(
        epochs=500,
        archive_every=10
    )
)