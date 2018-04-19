'''Adversarially learned inference and Bi-GAN
'''

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable

from .classifier import classify
from .gan import get_P_expectation, get_Q_expectation
from .modules.densenet import FullyConnectedNet
from .utils import cross_correlation


logger = logging.getLogger('cortex.arch' + __name__)

mnist_encoder_args_ = dict(dim_h=64, batch_norm=True, f_size=5, pad=2, stride=2, min_dim=7, dim_out=1028)
mnist_decoder_args_ = dict(dim_h=64, batch_norm=True, f_size=4, pad=1, stride=2, n_steps=2)
convnet_encoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3, fully_connected_layers=[1028])
convnet_decoder_args_ = dict(dim_h=64, batch_norm=True, n_steps=3)


def setup(model=None, data=None, routines=None, **kwargs):
    data['noise_variables']['z'] = (data['noise_variables']['z'][0], model['dim_z'])
    routines['generator']['measure'] = routines['discriminator']['measure']


def discriminator_routine(data, models, losses, results, viz, measure=None, boundary_seek=False, penalty=None):
    X_P, T, Z_Q = data.get_batch('images', 'targets', 'z')
    encoder, decoder = models['generator']
    x_disc, z_disc, topnet = models['discriminator']

    X_Q = decoder(Z_Q, nonlinearity=F.tanh)
    Z_P = encoder(X_P)

    W_P = x_disc(X_P, nonlinearity=F.relu)
    U_P = z_disc(Z_P, nonlinearity=F.relu)
    P_samples = topnet(torch.cat([W_P, U_P], 1))
    W_Q = x_disc(X_Q, nonlinearity=F.relu)
    U_Q = z_disc(Z_Q, nonlinearity=F.relu)
    Q_samples = topnet(torch.cat([W_Q, U_Q], 1))

    Ep = get_P_expectation(P_samples, measure)
    Eq = get_Q_expectation(Q_samples, measure)
    difference = Ep - Eq

    losses.update(discriminator=-difference)
    if penalty:
        X = Variable(X_P.data.cuda(), requires_grad=True)
        Z = Variable(Z_P.data.cuda(), requires_grad=True)
        W = x_disc(X, nonlinearity=F.relu)
        U = z_disc(Z, nonlinearity=F.relu)
        S = topnet(torch.cat([W, U], 1))

        G = autograd.grad(outputs=S, inputs=[X, Z], grad_outputs=torch.ones(S.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        G = G.view(G.size()[0], -1)
        G = (G ** 2).sum(1).mean()

        losses['discriminator'] += penalty * G
        results['gradient penalty'] = G.data[0]

    results.update(Scores=dict(Ep=Ep.data[0], Eq=Eq.data[0]))
    results['{} distance'.format(measure)] = difference.data[0]
    viz.add_image(X_P, name='ground truth')
    viz.add_image(X_Q, name='generated')
    viz.add_histogram(dict(fake=Q_samples.view(-1).data, real=P_samples.view(-1).data), name='discriminator output')
    viz.add_scatter(Z_P, labels=T.data, name='latent values')


def generator_routine(data, models, losses, results, viz, measure=None):
    X_P, Z_Q = data.get_batch('images', 'z')
    encoder, decoder = models['generator']
    x_disc, z_disc, topnet = models['discriminator']

    X_Q = decoder(Z_Q, nonlinearity=F.tanh)
    Z_P = encoder(X_P)

    W_P = x_disc(X_P, nonlinearity=F.relu)
    U_P = z_disc(Z_P, nonlinearity=F.relu)
    P_samples = topnet(torch.cat([W_P, U_P], 1))
    W_Q = x_disc(X_Q, nonlinearity=F.relu)
    U_Q = z_disc(Z_Q, nonlinearity=F.relu)
    Q_samples = topnet(torch.cat([W_Q, U_Q], 1))

    Ep = get_P_expectation(P_samples, measure)
    Eq = get_Q_expectation(Q_samples, measure)
    difference = Ep - Eq

    losses.update(generator=difference)


def network_routine(data, models, losses, results, viz):
    X, Y = data.get_batch('images', 'targets')
    encoder = models['generator'][0]
    classifier, decoder2 = models['nets']
    Z_P = encoder(X)

    Z_t = Variable(Z_P.data.cuda(), requires_grad=False)
    X_d = decoder2(Z_t, nonlinearity=F.tanh)
    dd_loss = ((X - X_d) ** 2).sum(1).sum(1).sum(1).mean()
    classify(classifier, Z_P, Y, losses=losses, results=results, key='nets')
    losses['nets'] += dd_loss

    correlations = cross_correlation(Z_P, remove_diagonal=True)
    viz.add_heatmap(correlations.data, name='latent correlations')
    viz.add_image(X_d, name='Reconstruction')


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
    discriminator_args_ = {}
    discriminator_args_.update(**encoder_args_)
    discriminator_args_.update(fully_connected_layers=[], batch_norm=False)

    encoder = Encoder(shape, dim_out=dim_z, **encoder_args_)
    decoder = Decoder(shape, dim_in=dim_z, **decoder_args_)
    decoder2 = Decoder(shape, dim_in=dim_z, **decoder_args_)
    x_disc = Encoder(shape, dim_out=256, **discriminator_args_)
    z_disc = FullyConnectedNet(dim_z, dim_h=[dim_z], dim_out=dim_z)
    topnet = FullyConnectedNet(256 + dim_z, dim_h=[512, 128], dim_out=1, batch_norm=False)
    classifier = FullyConnectedNet(dim_z, dim_h=[64, 64], dim_out=dim_l, batch_norm=True, dropout=0.2)

    models.update(discriminator=(x_disc, z_disc, topnet), generator=(encoder, decoder), nets=(classifier, decoder2))


ROUTINES = dict(discriminator=discriminator_routine, generator=generator_routine, nets=network_routine)


DEFAULT_CONFIG = dict(
    data=dict(batch_size=dict(train=64, test=640),
              noise_variables=dict(z=('normal', 64))),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=1e-4,
        updates_per_model=dict(discriminator=1, generator=1, nets=1)
    ),
    model=dict(model_type='convnet', dim_z=64, encoder_args=None, decoder_args=None),
    routines=dict(discriminator=dict(measure='GAN', penalty=1.0),
                  generator=dict(),
                  nets=dict()),
    train=dict(
        epochs=500,
        archive_every=10
    )
)