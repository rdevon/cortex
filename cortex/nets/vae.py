'''Simple Variational Autoencoder model.
'''
from cortex.models.FullyConnectedNet import FullyConnectedNet
from cortex.nets.classifier import classify
from cortex.nets.utils import update_encoder_args, update_decoder_args

__author__ = 'R Devon Hjelm and Samuel Lavoie'
__author_email__ = 'erroneus@gmail.com'

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.autograd import Variable

LOGGER = logging.getLogger('cortex.arch' + __name__)

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
        encoded = self.encoder(x)
        encoded = F.relu(encoded)
        self.mu = self.mu_net(encoded)
        self.std = self.logvar_net(encoded).exp_()
        self.latent = self.reparametrize(self.mu, self.std)
        return self.decoder(self.latent, nonlinearity=nonlinearity)

# ROUTINES =============================================================================================================
# Each of these methods needs to take `data`, `models`, `losses`, `results`, and `viz`

def vae_routine(data, models, losses, results, viz, vae_criterion=F.mse_loss, beta_kld=1., **kwargs):
    X, Y, Z = data.get_batch('images', 'targets', 'z')
    vae_net = models.vae

    vae_criterion = kwargs.get('criterion', vae_criterion) # For old-style

    outputs = vae_net(X)
    outputs = F.tanh(outputs)
    gen = vae_net.decoder(Z)
    gen = F.tanh(gen)

    r_loss = vae_criterion(outputs, X, size_average=False) / X.size(0)
    kl = 0.5 * (vae_net.std ** 2 + vae_net.mu ** 2 - 2. * torch.log(vae_net.std) - 1.).sum(1).mean()

    losses.vae=(r_loss + beta_kld * kl)
    #correlations = cross_correlation(vae_net.mu, remove_diagonal=True)

    results.update(KL_divergence=kl.item())
    viz.add_image(outputs, name='reconstruction')
    viz.add_image(gen, name='generated')
    viz.add_image(X, name='ground truth')
    #viz.add_heatmap(correlations.data, name='latent correlations')
    viz.add_scatter(vae_net.mu.data, labels=Y.data, name='latent values')


def classifier_routine(data, models, losses, results, viz, classifier_criterion=nn.CrossEntropyLoss(), **kwargs):
    X, Y = data.get_batch('images', 'targets')
    vae_net = models.vae
    classifier = models.classifier

    classifier_criterion = kwargs.get('criterion', classifier_criterion)  # For old-style

    vae_net(X)
    classify(classifier, vae_net.mu, Y, losses=losses, results=results, criterion=classifier_criterion)

# Building helper functions for autoencoders ===========================================================================


def build_encoder(models, x_shape, dim_out, Encoder, key='encoder', **encoder_args):
    LOGGER.debug('Forming encoder with class {} and args: {}'.format(Encoder, encoder_args))
    encoder = Encoder(x_shape, dim_out=dim_out, **encoder_args)

    if models is not None:
        models[key] = encoder

    return encoder


def build_decoder(models, x_shape, dim_in, Decoder, key='decoder', **decoder_args):
    LOGGER.debug('Forming dencoder with class {} and args: {}'.format(Decoder, decoder_args))
    decoder = Decoder(x_shape, dim_in=dim_in, **decoder_args)

    if models is not None:
        models[key] = decoder

    return decoder

# CORTEX ===============================================================================================================
# Must include `BUILD`, `TRAIN_ROUTINES`, `DEFAULT_CONFIG`

def BUILD(data, models, encoder_type='convnet', decoder_type='convnet', dim_z=64, dim_encoder_out=1028, encoder_args={},
          decoder_args={}):
    x_shape = data.get_dims('x', 'y', 'c')
    dim_l = data.get_dims('labels')

    Encoder, encoder_args = update_encoder_args(x_shape, model_type=encoder_type, encoder_args=encoder_args)
    Decoder, decoder_args = update_decoder_args(x_shape, model_type=decoder_type, decoder_args=decoder_args)

    encoder = build_encoder(None, x_shape, dim_encoder_out, Encoder, **encoder_args)
    decoder = build_decoder(None, x_shape, dim_z, Decoder, **decoder_args)
    vae = VAE(encoder, decoder, dim_out=dim_encoder_out, dim_z=dim_z)

    classifier = FullyConnectedNet(dim_z, dim_h=[200, 200], dim_out=dim_l, batch_norm=True, dropout=0.2)
    models.update(vae=vae, classifier=classifier)


TRAIN_ROUTINES = dict(vae=vae_routine, classifier=classifier_routine)

INFO = dict(vae_criterion=dict(help='Reconstruction criterion.'),
            beta_kld=dict(help='Beta scaling for KL term in lower-bound.'),
            classifier_criterion=dict(help='Classifier criterion for additional classifier.'),
            model_type=dict(choices=['mnist', 'convnet', 'resnet'],
                            help='Model type.'),
            dim_z=dict(help='Latent dimension.'),
            dim_encoder_out=dict(help='Dimension of the final layer of the decoder before decoding to mu and log sigma.'),
)

DEFAULT_CONFIG = dict(
    data=dict(batch_size=dict(train=64, test=640),
              noise_variables=dict(z=dict(dist='normal', size=64))),
    optimizer=dict(optimizer='Adam', learning_rate=1e-4),
    train=dict(epochs=500, archive_every=10, save_on_lowest='losses.vae'))
